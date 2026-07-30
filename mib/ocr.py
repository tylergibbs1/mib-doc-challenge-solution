"""Raster OCR with damage-adaptive retries, tuned for 4-vCPU offline runtime.

Confident full-page scans are OCR'd from their independent embedded raster;
ambiguous image pages use a trap-masked composited grayscale render. Hidden
PDF objects remain distrust signals and can never edit native scan pixels.

Speed notes (verified against rapidocr-onnxruntime 1.4.4 source):
- `max_side_len` is the REAL detector-resolution cap; `det_limit_side_len`
  never downscales a large page. 1280 ≈ 0.4x the default 2000 cost.
- One ORT intra-op thread per worker process; parallelism comes from the
  process pool, not thread pools (which thrash 4 vCPUs).
- cls disabled (forms are upright; 90/180/270 handled by the retry ladder
  only on weak pages).
"""
import os
from contextlib import contextmanager
from contextvars import ContextVar

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

_OCR = None
_OCR_HQ = None
_SELECTED_VIEW_OBSERVER = ContextVar(
    "mib_selected_ocr_view_observer", default=None)


@contextmanager
def capture_selected_view(observer):
    """Observe the exact winning engine input without changing OCR's API.

    The worker is sequential today, while ``ContextVar`` also keeps the hook
    isolated if callers later introduce threads or nested OCR operations.
    """
    token = _SELECTED_VIEW_OBSERVER.set(observer)
    try:
        yield
    finally:
        _SELECTED_VIEW_OBSERVER.reset(token)


def _notify_selected_view(image, preprocess, rotation_degrees):
    """Best-effort diagnostic callback; failures can never change OCR text."""
    try:
        observer = _SELECTED_VIEW_OBSERVER.get()
        if observer is not None:
            observer(image, preprocess, rotation_degrees)
    except Exception:
        pass


def _inject_session_failure_for_test():
    """Deterministic test seam for a transient ONNX/session failure.

    run_shard.py supplies the active case and extraction attempt.  Production
    is unaffected unless the explicitly test-only case variable is present.
    By default only attempt 1 fails, allowing the fresh-process retry to prove
    that it does not inherit the poisoned recognizer/session state.
    """
    target = os.environ.get("MIB_TEST_OCR_SESSION_FAIL_CASE")
    active = os.environ.get("MIB_ACTIVE_CASE")
    attempt = os.environ.get("MIB_EXTRACTION_ATTEMPT", "1")
    fail_attempt = os.environ.get("MIB_TEST_OCR_SESSION_FAIL_ATTEMPT", "1")
    if target and active == target and attempt == fail_attempt:
        raise RuntimeError("injected_onnx_session_failure")


def _rec_override():
    """Recognition-model selection (MIB_REC_MODEL env, else models/ artifact).

    The shipped recognizer is en_PP-OCRv5_mobile (English-specialized dict):
    measured +2.55 total on a 120-case dev sample against the bundled
    ch_PP-OCRv4 mobile, at equal-or-better speed. The detector stays PP-OCRv4
    mobile. Swappable because rapidocr reads the char dict from ONNX metadata,
    so any RapidAI-zoo rec model is a drop-in."""
    p = os.environ.get("MIB_REC_MODEL")
    if not p:
        from pathlib import Path
        cand = Path(__file__).resolve().parents[1] / "models" / "en_PP-OCRv5_rec_mobile.onnx"
        p = str(cand) if cand.exists() else None
    return {"rec_model_path": p} if p else {}


def _engine():
    global _OCR
    if _OCR is None:
        import cv2
        cv2.setNumThreads(1)
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR(
            intra_op_num_threads=1,
            inter_op_num_threads=1,
            use_cls=False,
            max_side_len=1280,
            det_limit_type="min",
            det_limit_side_len=736,
            **_rec_override(),
        )
    return _OCR


def _engine_hq():
    """Full-resolution engine for the escalation ladder (hard-decile pages).

    The 6s/PDF budget is an average: clean packets take ~0.9s on the fast
    engine, so packets with unread deny-relevant fields can afford seconds of
    high-quality re-OCR."""
    global _OCR_HQ
    if _OCR_HQ is None:
        import cv2
        cv2.setNumThreads(1)
        from rapidocr_onnxruntime import RapidOCR
        _OCR_HQ = RapidOCR(
            intra_op_num_threads=1,
            inter_op_num_threads=1,
            use_cls=False,
            max_side_len=2000,
            det_limit_type="min",
            det_limit_side_len=960,
            **_rec_override(),
        )
    return _OCR_HQ


def _stretch_faint(img):
    """Map [faint-ink .. paper] to full range for washed-out pages."""
    paper = float(np.median(img))
    lo = max(paper - 60.0, 1.0)
    hi = max(paper - 2.0, lo + 1.0)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _pepper_density(img):
    """Fraction of ALL pixels that are isolated dark specks. Clean scan pages
    measure <= 0.005; salt-and-pepper noise measures >= 0.020 (4x separation
    on rendered training pages)."""
    import cv2
    dark = (img < 100).astype(np.uint8)
    opened = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return int((dark & ~opened).sum()) / img.size


def ocr_page(img, min_lines=4, hq=False, force_stretch=False):
    """Return list of (text, confidence).

    A peppered page is median-filtered BEFORE the first pass (detection is
    upfront because pepper degrades reads without dropping below the retry
    gate — the ladder would never fire). Resurrection-safe by construction:
    hidden-span regions were painted paper-median before this image existed,
    so no filter can bring them back.

    Retry ladder (weak pages only, capped): faint-band contrast stretch, then
    90/270 rotations, then 180 as a last resort. Keeps the best pass.
    """
    _inject_session_failure_for_test()
    engine = _engine_hq() if hq else _engine()

    peppered = _pepper_density(img) > 0.011
    if peppered:
        import cv2
        # Surgical de-speckle for natively-peppered scans: repaint only
        # isolated dark pixels to paper (a blanket median filter erodes thin
        # glyph strokes — measured WORSE on the perturbation suite). Note the
        # honest scope: pepper that arrives through a resample (re-rendered
        # rasters) smears into gray mottle below this detector's threshold and
        # is an engine-level weakness (-7.5 on the every-page-peppered worst
        # case); the planned damage-trained recognizer is the real fix. This
        # branch is cheap insurance for direct-speck scans only.
        dark = (img < 100).astype(np.uint8)
        opened = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        isolated = (dark & ~opened).astype(bool)
        img = img.copy()
        img[isolated] = int(np.median(img))

    def run(image):
        from .forensics import sanitize_text
        result, _ = engine(image, use_cls=False)
        return [(sanitize_text(r[1].strip()), float(r[2]))
                for r in (result or []) if r[1].strip()]

    force_hq_stretch = hq and force_stretch
    if force_hq_stretch:
        img = _stretch_faint(img)
    base_preprocess = (
        "pepper_filter_stretch" if peppered and force_hq_stretch
        else "forced_hq_stretch" if force_hq_stretch
        else "pepper_filter" if peppered
        else "none"
    )
    best_image = img
    best_preprocess = base_preprocess
    best_rotation = 0
    best = run(best_image)
    if len(best) >= min_lines:
        _notify_selected_view(
            best_image, best_preprocess, best_rotation)
        return best

    stretched = img if force_hq_stretch else _stretch_faint(img)
    stretch_preprocess = ("pepper_filter_stretch" if peppered else "stretch")
    rotate_preprocess = ("pepper_filter_rotate" if peppered else "rotate")
    stretch_rotate_preprocess = (
        "pepper_filter_stretch_rotate" if peppered else "stretch_rotate")
    for candidate, preprocess, rotation in (
            (stretched, stretch_preprocess, 0),
            (np.rot90(img, 1), rotate_preprocess, 90),
            (np.rot90(stretched, 1), stretch_rotate_preprocess, 90),
            (np.rot90(stretched, 3), stretch_rotate_preprocess, 270)):
        lines = run(candidate)
        if len(lines) > len(best):
            best = lines
            best_image = candidate
            best_preprocess = preprocess
            best_rotation = rotation
        if len(best) >= min_lines:
            _notify_selected_view(
                best_image, best_preprocess, best_rotation)
            return best
    if len(best) < 2:
        candidate = np.rot90(stretched, 2)
        lines = run(candidate)
        if len(lines) > len(best):
            best = lines
            best_image = candidate
            best_preprocess = stretch_rotate_preprocess
            best_rotation = 180
    _notify_selected_view(best_image, best_preprocess, best_rotation)
    return best
