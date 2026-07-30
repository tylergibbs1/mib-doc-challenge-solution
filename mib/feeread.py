"""Narrow, asymmetric fee-status reader for damaged scanned receipts.

The shipped pixel decoder (``mib/pixmatch.py``) abstains on ``fee_status``: on
the eval half of dev the whole-string NCC decode measured 83% precision, below
the 95% ship bar, because "paid" aliases onto the tail of "unpaid" (substring
match) and washed-out receipts collapse the value margins. Those two failures
matter asymmetrically — a resurrected "unpaid" is a false denial and a
resurrected "paid" is a false approval — so this channel reads the fee line
with direction-specific evidence and refuses anything it cannot positively
support.

Design (validated on dev, 288 fee-unread cases):

* Receipt-page gate. Only pages the parser typed ``fee_receipt`` (pipeline-
  confirmed) or ``unknown`` with a strong pixmatch "Fee Status:" anchor
  (>= ``RECEIPT_LNCC``) are considered. This recovers damaged receipts the
  typer dropped to ``unknown`` (e.g. MIB-000192) without trusting phantom
  anchors that the same template throws on intake/registry pages.
* Cancellation-stamp guard. An office stamp (ARCHIVE / FILED / COPY /
  DUPLICATE / VOID / ...) marks a superseded copy whose printed value is not
  current — the parser already treats these pages as non-evidence, so the
  reader must too, or it re-introduces exactly the stale values the pipeline
  drops (e.g. MIB-000323 prints an unstruck "unpaid" under an ARCHIVE stamp
  while the true status is paid). "SAMPLE DENIAL" is a denial watermark, not a
  cancellation, and does not block the read.
* Prefix asymmetry. "unpaid" is accepted only when the "un" strokes are
  positively present (the "paid" template matches shifted right by the prefix
  width, with ink filling the [0, x) zone, and the whole-"unpaid" template
  corroborates). "paid" is accepted only when that prefix zone is positively
  clean (the "paid" template sits at the strip head with no ink to its left).
* Strike guard. A value cancelled by a colored vector strike is already in
  ``struck_values``; the reader drops it too, and ``decide`` re-filters the
  injected candidate for defense in depth.

Accepted reads are injected into the fee_status pool as
``[value, "fee_roi", 5, score, value]``; ``decide`` applies the usual
precedence and strike filtering.
"""
import os

from . import ocr, pixmatch

# Below this "Fee Status:" anchor strength an unknown-typed page is not trusted
# to be a receipt at all (dev: genuine unknown-typed receipts >= 0.81, the
# strongest intake phantom that aliases the label reads 0.774).
RECEIPT_LNCC = 0.80
# Anchor floor for even looking at a page (cheap pre-OCR filter).
ANCHOR_MIN = 0.55

# Office stamps that mark a receipt copy as non-current. INTAKE is a routing
# mark that rides on live intake pages and is deliberately excluded.
CANCEL_STAMPS = ("ARCHIVE", "FILED", "COPY", "DUPLICATE", "VOID", "SUPERSEDED",
                 "SUPERSED", "CANCELLED", "CANCELED", "DRAFT", "REISSUED",
                 "REPLACED", "EXPIRED", "SPECIMEN")

# paid acceptance floors: pipeline-confirmed receipts may read a little weaker
# than the damage-robust unknown-typed recoveries, which must clear a higher
# bar to exclude phantom anchors.
PAID_NCC_FEE_RECEIPT = 0.60
PAID_NCC_UNKNOWN = 0.70
PAID_X_MAX = 6                # "paid" must sit at the strip head
PAID_INK_LEFT_MAX = 1.5       # prefix zone must be clean
PAID_MARGIN = 0.06            # over the best non-paid value template

# unpaid acceptance: "paid" substring shifted right by the "un" prefix. The
# shift must be about one "un" width (~11px at 6.7pt): a true unpaid reads
# paid_x=11, while washed "waived"/leading-smudge aliases cluster at x>=14
# (measured on dev), so the upper bound is independent defense against them.
UNPAID_PAID_NCC = 0.80        # the shifted "paid" match must be strong
UNPAID_WHOLE_NCC = 0.50       # whole-"unpaid" template corroboration
UNPAID_X_MIN = 8              # "paid" shifted at least one "un" width
UNPAID_X_MAX = 13
UNPAID_INK_MIN = 1.5          # "un" strokes fill the [0, paid_x) zone
UNPAID_HEAD_DROP = 0.05       # shifted "paid" beats forced-at-head "paid"


def enabled():
    return os.environ.get("MIB_FEE_ROI", "1") != "0"


def _page_features(desk):
    """Value-line + receipt-identity features for one deskewed scan page, or
    None when there is no plausible "Fee Status:" anchor."""
    anchor = pixmatch.find_label(desk, pixmatch.FIELD_LABELS["fee_status"])
    if anchor is None or anchor[0] < ANCHOR_MIN:
        return None
    label = anchor[5]
    strip = pixmatch._value_strip(desk, anchor)

    def match(text, x_limit=None):
        return pixmatch._match(
            strip, pixmatch._value_tmpls(label, text), x_limit=x_limit)

    paid = match("paid")
    paid_head = match("paid", 8)
    unpaid = match("unpaid", 8)
    waived = match("waived", 8)
    unknown = match("unknown", 8)

    import cv2
    _, binc = cv2.threshold(
        strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_ink = (binc > 0).astype(float).mean(axis=0)
    paid_x = int(paid[1])
    ink_left_paid = float(col_ink[:paid_x].sum()) if paid_x > 0 else 0.0

    lines = ocr.ocr_page(desk, min_lines=1)
    joined = " ".join(t for t, _ in lines)
    upper = joined.upper()
    return {
        "label_ncc": float(anchor[0]),
        "paid_ncc": float(paid[0]), "paid_x": paid_x,
        "paid_head_ncc": float(paid_head[0]),
        "unpaid_ncc": float(unpaid[0]), "waived_ncc": float(waived[0]),
        "unknown_ncc": float(unknown[0]), "ink_left_paid": ink_left_paid,
        "stamps": [s for s in CANCEL_STAMPS if s in upper],
    }


def _classify(feat, page_type, struck):
    """Return an accepted fee value ('paid'/'unpaid') for one page, or None."""
    is_receipt = (page_type == "fee_receipt"
                  or (page_type == "unknown"
                      and feat["label_ncc"] >= RECEIPT_LNCC))
    if not is_receipt or feat["stamps"]:
        return None
    p = feat["paid_ncc"]
    px = feat["paid_x"]
    il = feat["ink_left_paid"]
    # UNPAID
    if (p >= UNPAID_PAID_NCC and UNPAID_X_MIN <= px <= UNPAID_X_MAX
            and il >= UNPAID_INK_MIN
            and feat["unpaid_ncc"] >= UNPAID_WHOLE_NCC
            and p > feat["waived_ncc"] and p > feat["unknown_ncc"]
            and feat["paid_head_ncc"] <= p - UNPAID_HEAD_DROP):
        if "unpaid" not in struck:
            return "unpaid"
    # PAID
    paid_floor = (PAID_NCC_UNKNOWN if page_type == "unknown"
                  else PAID_NCC_FEE_RECEIPT)
    if (p >= paid_floor and px <= PAID_X_MAX and il <= PAID_INK_LEFT_MAX
            and feat["paid_head_ncc"] >= paid_floor - 0.02):
        runner = max(feat["waived_ncc"], feat["unknown_ncc"],
                     feat["unpaid_ncc"])
        if p - runner >= PAID_MARGIN and "paid" not in struck:
            return "paid"
    return None


def read_fee_status(doc, page_types_by_no, struck_values=(),
                    hidden_spans=None):
    """Read the fee status from a damaged scanned receipt.

    Returns ``(value, score, provenance)`` for the strongest accepted receipt
    page, or None. ``page_types_by_no`` maps page number -> parser page type.
    """
    struck = {str(v).lower() for v in (struck_values or ())}
    best = None
    for page_no, img in pixmatch._p0b_scan_images(doc, hidden_spans):
        page_type = page_types_by_no.get(page_no)
        if page_type not in ("fee_receipt", "unknown"):
            continue
        desk, _ = pixmatch.deskew_robust(img)
        feat = _page_features(desk)
        if feat is None:
            continue
        value = _classify(feat, page_type, struck)
        if value is None:
            continue
        if best is None or feat["label_ncc"] > best[2]["label_ncc"]:
            score = round(min(97.0, feat["paid_ncc"] * 100 + 5), 1)
            best = (value, score, feat, page_no)
    if best is None:
        return None
    value, score, feat, page_no = best
    return value, score, {"page": page_no, "label_ncc": round(
        feat["label_ncc"], 4), "value": value}


def fee_roi_candidate(doc, page_types_by_no, struck_values=(),
                      hidden_spans=None):
    """Pool candidate ``[value, "fee_roi", 5, score, value]`` or None."""
    if not enabled():
        return None
    read = read_fee_status(doc, page_types_by_no, struck_values, hidden_spans)
    if read is None:
        return None
    value, score, _prov = read
    return [value, "fee_roi", 5, score, value]
