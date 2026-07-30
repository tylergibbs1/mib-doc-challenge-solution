"""Constrained-candidate scoring against the shipped CTC recognizer.

Instead of decoding the recognizer's argmax string and edit-distance-snapping
it to the vocabulary, score every legal candidate value directly against the
model's per-frame posteriorgram with the CTC forward algorithm. This uses the
full posterior (a garbled argmax often hides a clean second-choice path) and
yields exact P(candidate | image) up to normalization — a second evidence
channel whose failure modes are independent of template correlation.

Runs the same en_PP-OCRv5 rec ONNX the OCR engine ships; no extra artifact.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

_SESSION = None
_CHARS = None
_CHAR_TO_IX = None

_REC_SHAPE = (48, 320)      # PP-OCR rec input (H, max W)
_NEG = -1e9


def _session():
    global _SESSION, _CHARS, _CHAR_TO_IX
    if _SESSION is None:
        import onnxruntime as ort
        p = os.environ.get("MIB_REC_MODEL") or str(
            Path(__file__).resolve().parents[1] / "models" / "en_PP-OCRv5_rec_mobile.onnx")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        _SESSION = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        chars = [c for c in
                 _SESSION.get_modelmeta().custom_metadata_map["character"].split("\n")
                 if c != ""]
        # CTC head layout: [blank] + dict + [space]
        _CHARS = ["<blank>"] + chars + [" "]
        _CHAR_TO_IX = {c: i for i, c in enumerate(_CHARS)}
    return _SESSION


def _preprocess(gray):
    """Grayscale line strip -> the rec model's 3x48xW normalized tensor."""
    import cv2
    h, w = gray.shape
    tw = min(_REC_SHAPE[1], max(8, int(round(_REC_SHAPE[0] * w / h))))
    img = cv2.resize(gray, (tw, _REC_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    if tw < _REC_SHAPE[1]:
        x = np.pad(x, ((0, 0), (0, _REC_SHAPE[1] - tw)), constant_values=0.0)
    return np.repeat(x[None, None], 3, axis=1)


def posteriorgram(gray):
    """(T, C) log-probability frames for one line strip."""
    sess = _session()
    y = sess.run(None, {sess.get_inputs()[0].name: _preprocess(gray)})[0][0]
    return np.log(np.maximum(y, 1e-12))


@lru_cache(maxsize=8192)
def _label_ixs(text):
    """Candidate string -> tuple of class indices, or None if out-of-dict."""
    _session()
    out = []
    for ch in text:
        ix = _CHAR_TO_IX.get(ch)
        if ix is None:
            return None
        out.append(ix)
    return tuple(out)


def _ctc_logp(logp, labels):
    """Log P(labels | frames) by the CTC forward recursion."""
    T = logp.shape[0]
    ext = [0]
    for l in labels:
        ext += [l, 0]
    S = len(ext)
    if S > 2 * T + 1:
        return _NEG
    alpha = np.full(S, _NEG)
    alpha[0] = logp[0, 0]
    if S > 1:
        alpha[1] = logp[0, ext[1]]
    for t in range(1, T):
        prev = alpha
        alpha = np.full(S, _NEG)
        for s in range(S):
            best = prev[s]
            if s >= 1:
                best = np.logaddexp(best, prev[s - 1])
            if s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2]:
                best = np.logaddexp(best, prev[s - 2])
            alpha[s] = best + logp[t, ext[s]]
    tail = alpha[-1] if S == 1 else np.logaddexp(alpha[-1], alpha[-2])
    return float(tail)


def greedy_decode(gray):
    """Best-path (argmax-collapse) string for one line strip."""
    logp = posteriorgram(gray)
    ixs = logp.argmax(axis=1)
    out, prev = [], 0
    for ix in ixs:
        if ix != 0 and ix != prev:
            out.append(_CHARS[ix])
        prev = ix
    return "".join(out)


def score(gray, candidates):
    """Rank candidate strings by length-normalized CTC log-probability.

    Returns [(norm_logp, candidate)] best-first; candidates containing
    characters outside the model dictionary are skipped."""
    logp = posteriorgram(gray)
    out = []
    for cand in candidates:
        ixs = _label_ixs(cand)
        if ixs is None:
            continue
        lp = _ctc_logp(logp, ixs)
        out.append((lp / max(1, len(ixs)), cand))
    out.sort(reverse=True)
    return out
