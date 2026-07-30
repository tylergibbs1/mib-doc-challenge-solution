"""Closed-vocabulary pixel decoder for damaged scan pages.

Scan pages are the generator's compact form template rasterized once (72 DPI),
upscaled 2x to 1224x1584, damaged (small rotation, washout, pepper, smudges),
and JPEG-compressed. Because every target field takes values from a small
legal set, degraded reading becomes hypothesis scoring rather than open
recognition: correlate a template of each legal value against the page
(Kopec & Chou's document image decoding — under i.i.d. speckle the matched
filter is the ML decoder, and NCC's gain/offset normalization makes washout
largely invisible to the ranking).

Templates are EMPIRICAL where possible: models/pix_bank.npz holds real line
crops harvested from clean scan pages (OCR text == label + training-truth
value, so each crop is verified twice), split into label/value parts. Real-vs-
real matching carries the generator's exact rasterizer + JPEG response that
synthetic renders miss (~0.3 NCC of fidelity). Unbanked values fall back to
synthetic base-14 renders in the per-label weight/size measured on dev.

Injection-inert by construction: only the embedded scan raster is read (hidden
PDF text cannot exist inside a raster), and only legal vocabulary values can
be emitted. The enabled native experiment never lets PDF-object bounding boxes
modify independent scan pixels; the disabled control intentionally retains the
previously approved P0-B implementation for exact comparison. Reads enter the
evidence pools at harvest rank and pass the same precedence / agreement /
approval-gate machinery as every other source.
"""
import itertools
import hashlib
import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import forensics
from .vocab import FEES, FLAGS, PURPOSES, SPECIES, VISAS, WORLDS

LABEL_MIN_NCC = 0.45          # below this the label anchor is not trusted
VALUE_STRIP_W = 560           # px right of the label to search for the value
MIN_SCAN_W = 1100             # P0-B full-page scan signature (1224 nominal)

# field -> visible label variants on the scan template.
FIELD_LABELS = {
    "applicant_name": ("Applicant:", "Registry Name:"),
    "species_code": ("Species Code:", "Species Match:"),
    "home_world": ("Home World:",),
    "visa_class": ("Visa Class:",),
    "sponsor_id": ("Sponsor ID:",),
    "arrival_date": ("Arrival Date:",),
    "declared_purpose": ("Declared Purpose:",),
    "risk_flags": ("Observed flags:",),
    "fee_status": ("Fee Status:",),
    "registry_status": ("Registry Status:",),
}

# page types allowed to source each field (unknown page type allows all).
FIELD_PAGES = {
    "applicant_name": {"intake", "registry", "biometric"},
    "species_code": {"intake", "registry", "biometric"},
    "home_world": {"intake", "registry"},
    "visa_class": {"intake"},
    "sponsor_id": {"intake"},
    "arrival_date": {"intake", "registry"},
    "declared_purpose": {"intake"},
    "risk_flags": {"biometric"},
    "fee_status": {"fee_receipt"},
    "registry_status": {"registry"},
}

# synthetic-fallback style per label: (bold, size), measured per page type on
# dev by candidate discrimination (intake/registry/fee regular, biometric bold).
_STYLE = {"Species Match:": (True, 6.7), "Observed flags:": (True, 6.7),
          "Fee Status:": (False, 6.7), "Registry Status:": (False, 6.7),
          "Registry Name:": (False, 6.7)}
_DEFAULT_STYLE = (False, 6.5)

_ENUM_VALUES = {
    "species_code": SPECIES,
    "home_world": WORLDS,
    "visa_class": VISAS,
    "declared_purpose": PURPOSES,
    "fee_status": FEES,
    "registry_status": ("CLEAR", "EMBARGO REVIEW"),
}

# risk_flags prints "none" or a sorted pipe-joined set; singles + pairs cover
# the corpus (triples unobserved).
_FLAG_STRINGS = (("none",) + tuple(FLAGS)
                 + tuple("|".join(p) for p in itertools.combinations(sorted(FLAGS), 2)))

# Accept-gates per field, chosen on the eval half of dev at >=95% precision
# (tools/pixstudy.py / pixanalyze.py); tools/pixapply.py simulations mirror
# them. A field absent here never ships a read: on the eval half, visa (88%),
# purpose (89%), sponsor (71%), date (82%), fee (83%), flags (deny-direction
# never fires; approve-direction 62%) and name all measured below the bar —
# where OCR fails, the content is usually genuinely destroyed and the margins
# collapse, which is the channel abstaining correctly. "ctc" additionally
# requires the CTC channel (mib/ctcscore.py) to agree on the same strip —
# mandatory for approve-enabling values regardless of this table.
GATES = {
    "species_code": {"ncc": 0.45, "margin": 0.08},   # 96% @ n=95 eval-half
    "home_world": {"ncc": 0.55, "margin": 0.12},     # 96% @ n=57 eval-half
}

# values whose acceptance can only ever help an approval; they always require
# CTC agreement on top of their field gate.
APPROVE_ENABLING = {("fee_status", "paid"), ("fee_status", "waived"),
                    ("risk_flags", "none")}


def passes_gate(field, read):
    g = GATES.get(field)
    if g is None:
        return False
    if read["ncc"] < g["ncc"] or read["margin"] < g["margin"]:
        return False
    return True


def needs_ctc(field, value):
    g = GATES.get(field) or {}
    return bool(g.get("ctc")) or (field, str(value)) in APPROVE_ENABLING


def verify_ctc(field, strip, value):
    """Second-channel agreement check on the same strip (independent failure
    modes: learned CTC posterior vs template correlation)."""
    from . import ctcscore
    try:
        if field in ("sponsor_id", "arrival_date", "applicant_name"):
            greedy = ctcscore.greedy_decode(strip)
            return greedy.replace(" ", "") == str(value).replace(" ", "")
        cands = (_FLAG_STRINGS if field == "risk_flags"
                 else _ENUM_VALUES.get(field, ()))
        scored = ctcscore.score(strip, cands)
        return bool(scored) and scored[0][1] == value
    except Exception:
        return False


_BANK = None


def _bank():
    global _BANK
    if _BANK is None:
        p = os.environ.get("MIB_PIX_BANK") or str(
            Path(__file__).resolve().parents[1] / "models" / "pix_bank.npz")
        _BANK = {"l": {}, "v": {}, "d": {}}
        if Path(p).exists():
            z = np.load(p)
            for k in z.files:
                parts = k.split("|")
                if parts[0] == "l":
                    _BANK["l"].setdefault(parts[1], []).append(z[k])
                elif parts[0] == "v":
                    _BANK["v"].setdefault(parts[1], {}).setdefault(parts[2], []).append(z[k])
                elif parts[0] == "d":
                    _BANK["d"].setdefault(parts[1], []).append(z[k])
    return _BANK


def render_text(text, size, bold=False):
    """Synthetic fallback template: base-14 render on the 2x grid."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=900, height=60)
    page.insert_text((8, 40), text, fontname="hebo" if bold else "helv",
                     fontsize=size)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    doc.close()
    ys, xs = np.where(arr < 250)
    if len(ys) == 0:
        return None
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


@lru_cache(maxsize=8192)
def _syn(text, label):
    bold, size = _STYLE.get(label, _DEFAULT_STYLE)
    t = render_text(text, size, bold)
    return (t,) if t is not None else ()


def _notify_scan_view(observer, **view):
    """One-way immutable fingerprint hook; decoding ignores its result."""
    try:
        if observer is not None:
            image = np.ascontiguousarray(view.pop("image"))
            observer(
                **view,
                shape=[int(value) for value in image.shape],
                dtype=str(image.dtype),
                pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
    except Exception:
        pass


def _image_dpi(doc, page_number, image):
    try:
        page = doc[int(page_number)]
        return 72.0 * min(
            image.shape[1] / page.rect.width,
            image.shape[0] / page.rect.height)
    except Exception:
        return 72.0


def scan_images(doc, hidden_spans=None, visible_spans=None,
                view_observer=None, native_view=None):
    """[(page_number, native-res grayscale)] for full-page scan pages.

    When the native experiment is enabled, reads a confidently selected
    embedded scan directly with no resample blur and ignores ``hidden_spans``:
    PDF-object geometry must never erase independent scan ink. Disabled runs
    delegate to the exact P0-B implementation below.

    ``native_view`` selects the pixel channel explicitly: the baseline ledger
    passes ``False`` so its pixmatch reads stay the P0-B masked scan regardless
    of the process-level ``MIB_NATIVE_SCAN_OCR`` flag, and the native ledger
    passes ``True``. ``None`` falls back to the flag for tools/tests.
    """
    # Default-off must remain the independently approved P0-B behavior. The
    # hardened selector is part of the native two-view experiment; sharing it
    # with the off control could make selector drift invisible to a
    # base-versus-variant false-approval audit.
    if native_view is None:
        native_view = os.environ.get("MIB_NATIVE_SCAN_OCR", "1") == "1"
    if not native_view:
        images = _p0b_scan_images(doc, hidden_spans)
        for page_number, image in images:
            masked = any(
                getattr(span, "page", None) == int(page_number)
                for span in hidden_spans or ())
            _notify_scan_view(
                view_observer, page_number=int(page_number),
                transform="p0b_scan_output", image=image,
                source="p0b_masked_scan_image",
                dpi=_image_dpi(doc, page_number, image),
                rotation_degrees=0.0,
                preprocess=("grayscale_despeckle_hidden_mask" if masked
                            else "grayscale_despeckle"))
        return images

    out = []
    for page in doc:
        img, provenance = forensics.native_scan_gray(
            doc, page, visible_spans=visible_spans,
            view_observer=view_observer)
        if img is not None:
            processed = despeckle(img)
            _notify_scan_view(
                view_observer, page_number=int(page.number),
                transform="despeckled", image=processed,
                source="native_full_page_image",
                dpi=float(provenance["output_dpi"]),
                rotation_degrees=0.0, preprocess="despeckle")
            out.append((page.number, processed))
    return out


def _p0b_scan_images(doc, hidden_spans=None):
    """Byte-for-byte P0-B pixel-channel selection for the disabled control."""
    import io

    import cv2
    from PIL import Image

    out = []
    for page in doc:
        ims = [im for im in page.get_images(full=True) if im[2] >= MIN_SCAN_W]
        if not ims:
            continue
        info = doc.extract_image(ims[0][0])
        img = despeckle(np.array(Image.open(io.BytesIO(info["image"])).convert("L")))
        if hidden_spans:
            sx = img.shape[1] / page.rect.width
            sy = img.shape[0] / page.rect.height
            paper = int(np.median(img))
            for span in hidden_spans:
                if span.page != page.number:
                    continue
                x0, y0, x1, y1 = span.bbox
                cv2.rectangle(img, (int(x0 * sx) - 2, int(y0 * sy) - 2),
                              (int(x1 * sx) + 2, int(y1 * sy) + 2), paper, -1)
        out.append((page.number, img))
    return out


def despeckle(img):
    """Surgical de-speckle (same logic as ocr.ocr_page): isolated dark pixels
    inflate every NCC window's variance and crush correlation — the matched
    filter is robust to what pepper does to the signal, but the normalization
    denominator is not. Strokes survive the isolation test."""
    import cv2
    dark = (img < 100).astype(np.uint8)
    opened = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    isolated = (dark & ~opened).astype(bool)
    if isolated.any():
        img = img.copy()
        img[isolated] = int(np.median(img))
    return img


def deskew(img):
    """Undo the simulated crooked-scanner rotation (±~4°).

    The form frame and table rules are long straight lines; the median angle
    of near-horizontal Hough segments recovers the skew to ~0.1°, well inside
    NCC's ~0.5° tolerance for 7pt strips."""
    import cv2
    edges = cv2.Canny(img, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=120,
                            minLineLength=350, maxLineGap=4)
    if lines is None:
        return img, 0.0
    angles = []
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) < 200:
            continue
        a = np.degrees(np.arctan2(dy, dx))
        if abs(a) <= 8.0:
            angles.append(a)
    if not angles:
        return img, 0.0
    angle = float(np.median(angles))
    if abs(angle) < 0.15:
        return img, 0.0
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=int(np.median(img))), angle


def deskew_robust(img):
    """deskew(), with an ink-orientation fallback for degraded pages.

    Hough needs the long form rules; heavy wash/pepper erodes them below the
    line threshold and deskew() silently returns 0.0 on exactly the pages
    that need it (verified: a truly ~2.4-degree page reported 0.0). Fallback:
    dilate ink into text-line blobs and take the median minAreaRect angle,
    applied only in the 0.4-6 degree band so clean pages and noise estimates
    never trigger a rotation. Used by the ROI readers (fee/note), whose
    NCC templates lose margin beyond ~0.5 degrees; the pixmatch channel
    keeps plain deskew() until its gates are re-measured under the fallback."""
    import cv2
    desk, ang = deskew(img)
    if abs(ang) >= 0.15:
        return desk, ang
    th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    dil = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)),
                     iterations=1)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angs = []
    for c in cnts:
        if cv2.contourArea(c) < 800:
            continue
        (_, _), (bw, bh), a = cv2.minAreaRect(c)
        if bw < bh:
            a += 90
        while a > 45:
            a -= 90
        while a < -45:
            a += 90
        if abs(a) <= 8:
            angs.append(a)
    if len(angs) < 4:
        return desk, ang
    ra = float(np.median(angs))
    if not 0.4 <= abs(ra) <= 6.0:
        return desk, ang
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ra, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=int(np.median(img))), ra


def _match(img, tmpls, x_limit=None):
    """(best_ncc, x, y, w, h) over a list/tuple of template instances.

    x_limit restricts the match START column — the search region is cropped
    per template so left-anchored values cannot alias onto substrings further
    right (e.g. "paid" onto the tail of "unpaid")."""
    import cv2
    best = (-1.0, 0, 0, 0, 0)
    for tmpl in tmpls or ():
        sub = img if x_limit is None else img[:, :x_limit + tmpl.shape[1]]
        if (tmpl is None or sub.shape[0] < tmpl.shape[0]
                or sub.shape[1] < tmpl.shape[1]):
            continue
        r = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (float(mx), int(loc[0]), int(loc[1]),
                    tmpl.shape[1], tmpl.shape[0])
    return best


def _label_tmpls(label):
    return tuple(_bank()["l"].get(label, ())) + _syn(label + " ", label)


def _value_tmpls(label, value):
    return (tuple(_bank()["v"].get(label, {}).get(value, ()))
            + _syn(str(value), label))


def find_label(img, labels, region=None):
    """Best label anchor across variants: (ncc, x, y, w, h, label) or None."""
    y0, y1, x0, x1 = region or (100, img.shape[0], 0, 560)
    sub = img[y0:y1, x0:x1]
    best = None
    for text in labels:
        ncc, x, y, w, h = _match(sub, _label_tmpls(text))
        if best is None or ncc > best[0]:
            best = (ncc, x + x0, y + y0, w, h, text)
    if best is None or best[0] < LABEL_MIN_NCC:
        return None
    return best


def _value_strip(img, anchor):
    ncc, lx, ly, lw, lh, label = anchor
    y0 = max(0, ly - 4)
    y1 = min(img.shape[0], ly + lh + 5)
    x0 = lx + lw - 3
    x1 = min(img.shape[1], x0 + VALUE_STRIP_W)
    return img[y0:y1, x0:x1]


def _rank(strip, candidates, label, x_max=None, uniform=False):
    """Rank candidates by NCC. `uniform=True` scores every candidate from the
    synthetic renderer only — mandatory for combinatorial fields (dates,
    names) where the bank covers some candidates and not others: mixed
    template quality would rank by bank membership instead of image content
    (real-crop templates score ~0.25 higher than synthetic on real pixels)."""
    scored = []
    for cand in candidates:
        tmpls = _syn(cand, label) if uniform else _value_tmpls(label, cand)
        ncc, _, _, _, _ = _match(strip, tmpls, x_limit=x_max)
        scored.append((ncc, cand))
    scored.sort(reverse=True)
    return scored


def _decode_enum(strip, values, label):
    # Values are left-aligned right after the label, so matches must sit at
    # the strip head: without the anchor, "paid" slides onto the tail of
    # "unpaid" (substring aliasing) and short values match junk anywhere in
    # the strip.
    scored = _rank(strip, values, label, x_max=30)
    if not scored or scored[0][0] <= 0:
        return None
    best, runner = scored[0], (scored[1] if len(scored) > 1 else (0.0, ""))
    return {"value": best[1], "ncc": round(best[0], 4),
            "margin": round(best[0] - runner[0], 4)}


def _decode_name(strip, lexicon, label):
    """Two-stage joint decode over the 144x144 name grammar. The first token
    must sit at the strip head (the value is left-aligned after the label) —
    without that constraint short tokens match spuriously anywhere."""
    firsts = _rank(strip, tuple(lexicon["first"]), label, x_max=30, uniform=True)[:5]
    if not firsts or firsts[0][0] <= 0:
        return None
    best = []
    for fncc, ftok in firsts:
        _, fx, _, fw, _ = _match(strip, _syn(ftok, label), x_limit=30)
        sub = strip[:, fx + fw + 1:fx + fw + 240]
        lasts = _rank(sub, tuple(lexicon["last"]), label, x_max=14, uniform=True)[:2]
        for lncc, ltok in lasts:
            if lncc > 0:
                best.append(((fncc + lncc) / 2, f"{ftok} {ltok}"))
    if not best:
        return None
    best.sort(reverse=True)
    runner = best[1][0] if len(best) > 1 else 0.0
    return {"value": best[0][1], "ncc": round(best[0][0], 4),
            "margin": round(best[0][0] - runner, 4)}


def _digit_tmpls(ch, label):
    return tuple(_bank()["d"].get(ch, ())) + _syn(ch, label)


def _prefix_tmpls(label, prefix, frac):
    """Anchor instances for a printed value prefix: the front `frac` of every
    banked value crop under this label (real pixels), plus synthetic."""
    crops = []
    for insts in _bank()["v"].get(label, {}).values():
        for c in insts[:2]:
            w = int(c.shape[1] * frac)
            if w >= 8:
                crops.append(c[:, :w])
    return tuple(crops[:10]) + _syn(prefix, label)


@lru_cache(maxsize=16)
def _advance(size):
    """Digit advance in scan pixels (Helvetica digits are tabular)."""
    return render_text("00", size).shape[1] - render_text("0", size).shape[1]


def _metric_frac(text, i):
    import fitz
    total = fitz.get_text_length(text, fontname="helv", fontsize=10)
    return fitz.get_text_length(text[:i], fontname="helv", fontsize=10) / total


def _classify_slots(band, x, w, text, slot_sets, label):
    """Classify character slots of an anchored skeleton at font-metric
    offsets. Returns (chars, min_ncc, min_margin) or None."""
    chars = list(text)
    nccs, margins = [], []
    for i, legal in slot_sets.items():
        f0 = _metric_frac(text, i)
        f1 = _metric_frac(text, i + 1)
        w0 = x + int(w * f0) - 2
        w1 = x + int(w * f1) + 3
        win = band[:, max(0, w0):w1]
        scored = sorted(((_match(win, _digit_tmpls(d, label))[0], d)
                         for d in legal), reverse=True)
        if not scored or scored[0][0] <= 0:
            return None
        chars[i] = scored[0][1]
        nccs.append(scored[0][0])
        margins.append(scored[0][0] - scored[1][0] if len(scored) > 1 else scored[0][0])
    return "".join(chars), min(nccs), min(margins)


def _decode_sponsor(strip, label):
    """SPN-#### via the printed prefix as anchor, then per-slot digit
    classification at the font's uniform digit advance."""
    pncc, px, py, pw, ph = _match(strip, _prefix_tmpls(label, "SPN-", 0.5),
                                  x_limit=30)
    if pncc < 0.30:
        return None
    syn = _syn("SPN-", label)
    scale = pw / syn[0].shape[1] if syn else 1.0
    pitch = max(4, int(round(_advance(_STYLE.get(label, _DEFAULT_STYLE)[1]) * scale)))
    band = strip[max(0, py - 3):py + ph + 4]
    digits, nccs, margins = [], [], []
    x = px + pw
    for slot in range(4):
        w0 = max(0, x - 2 + slot * pitch)
        win = band[:, w0:w0 + pitch + 6]
        scored = sorted(((_match(win, _digit_tmpls(d, label))[0], d)
                         for d in "0123456789"), reverse=True)
        if scored[0][0] <= 0:
            return None
        digits.append(scored[0][1])
        nccs.append(scored[0][0])
        margins.append(scored[0][0] - scored[1][0])
    return {"value": "SPN-" + "".join(digits),
            "ncc": round(min(nccs), 4), "margin": round(min(margins), 4),
            "prefix_ncc": round(pncc, 4)}


# uncertain slots of "YYYY-MM-DD" and their legal digit sets (year prefix
# "20" is verified by the skeleton anchor itself)
_DATE_SLOTS = {2: "0123456789", 3: "0123456789", 5: "01",
               6: "0123456789", 8: "0123", 9: "0123456789"}


def _decode_date(strip, label, lo=date(2024, 6, 1), hi=date(2026, 12, 31)):
    """Two-stage date read: a coarse full-string scan (uniform synthetic
    templates; the hyphens anchor the skeleton) followed by per-slot digit
    classification with the empirical atlas at font-metric offsets — the
    coarse stage nails alignment, the slots supply single-digit
    discrimination that whole-string NCC lacks at 7pt."""
    cands = []
    d = lo
    while d <= hi:
        cands.append(d.isoformat())
        d += timedelta(days=1)
    coarse = _rank(strip, tuple(cands), label, x_max=30, uniform=True)
    if not coarse or coarse[0][0] <= 0:
        return None
    text = coarse[0][1]
    ncc0, x, y, w, h = _match(strip, _syn(text, label), x_limit=30)
    band = strip[max(0, y - 3):y + h + 4]
    refined = _classify_slots(band, x, w, text, _DATE_SLOTS, label)
    cmargin = coarse[0][0] - coarse[1][0]
    if refined:
        composed, sncc, smargin = refined
        try:
            if lo <= date.fromisoformat(composed) <= hi:
                return {"value": composed, "ncc": round(sncc, 4),
                        "margin": round(smargin, 4),
                        "coarse_ncc": round(coarse[0][0], 4),
                        "coarse_value": text}
        except ValueError:
            pass
    return {"value": text, "ncc": round(coarse[0][0], 4),
            "margin": round(cmargin, 4)}


def decode_field(img, field, name_lexicon=None):
    """One field on one deskewed scan page. Returns read dict or None."""
    anchor = find_label(img, FIELD_LABELS[field])
    if anchor is None:
        return None
    strip = _value_strip(img, anchor)
    label = anchor[5]
    if field == "applicant_name":
        r = _decode_name(strip, name_lexicon, label) if name_lexicon else None
    elif field == "sponsor_id":
        r = _decode_sponsor(strip, label)
    elif field == "arrival_date":
        r = _decode_date(strip, label)
    elif field == "risk_flags":
        r = _decode_enum(strip, _FLAG_STRINGS, label)
    else:
        r = _decode_enum(strip, _ENUM_VALUES[field], label)
    if r:
        r["label_ncc"] = round(anchor[0], 4)
        r["label"] = label
        ncc, lx, ly, lw, lh, _ = anchor
        r["strip_box"] = [max(0, ly - 4), min(img.shape[0], ly + lh + 5),
                          lx + lw - 3, min(img.shape[1], lx + lw - 3 + VALUE_STRIP_W)]
    return r


def decode(images, fields, name_lexicon=None, page_types=None):
    """Decode `fields` across pre-deskewed scan page images.

    images: [(page_number, deskewed grayscale)]. page_types (optional) maps
    page_number -> parsed page type; a field is only sought on page types that
    carry it (unknown types allow everything). Best read per field by NCC."""
    out = {}
    for field in fields:
        allowed = FIELD_PAGES.get(field)
        best = None
        for pno, img in images:
            pt = (page_types or {}).get(pno)
            if pt == "foreign":
                continue
            if pt and allowed and pt not in allowed and pt != "unknown":
                continue
            r = decode_field(img, field, name_lexicon)
            if r and (best is None or r["ncc"] > best["ncc"]):
                r["page"] = pno
                best = r
        if best:
            out[field] = best
    return out
