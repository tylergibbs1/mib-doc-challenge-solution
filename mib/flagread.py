"""Narrow disqualifying-flag reader for damaged scan pages (deny-direction only).

The pixel decoder (``mib/pixmatch.py``) abstains on ``risk_flags`` entirely: on
the eval half the approve-direction read measured 62% and the deny-direction
never cleared the bar, because every flag value is a 13-17 character underscore
word and the templates alias heavily onto one another — a "biohazard_red"
template scores as high on a printed "illegible_biometrics" line as on a real
biohazard line (both long, same weight). A blur-tolerant flag-token correlator
was separately killed at 7-9% corpus precision for the same reason.

This channel reads ONLY the four DISQUALIFYING flags
``{memory_tampering, planetary_embargo, active_warrant, biohazard_red}`` and
fires only on positive, corroborated evidence, so a fire can only ADD a
disqualifying flag — which can only move a case toward denial, never toward
approval. It is therefore structurally incapable of creating a false approval.

Design (validated on dev, 701-case run population = every scan-bearing packet
whose ``risk_flags`` pool carries no disqualifying flag; census 2/2 fires
truth-correct, 0 fires on the 427 truth-none packets, 0 fires on the red-team
corpus):

* Anchored ROI. ``find_label`` over the flags label variants anchors the value
  strip on slip/intake/registry AND unknown-typed pages (unknown allows all).
* TWO INDEPENDENT VIEWS must agree on the same disqualifying flag:
  - Template correlation. The full-word template (bank + synthetic) must match
    at the strip head under BOTH raw and contrast-stretched preprocessings,
    and the flag's DISTINCTIVE underscore SUFFIX ("ard_red", "_embargo",
    "_warrant", "_tampering") must also match just right of the head. The
    suffix is what a "none"/short alias cannot satisfy.
  - Glyph recognition. The shipped CTC recognizer (``mib/ctcscore.py``), whose
    failure modes are independent of template correlation, must rank the SAME
    flag first among all flags (incl. "none"), above an absolute log-prob floor
    and by a wide margin over "none". This is the only feature that separates a
    real disqualifying line from a printed review-flag line
    (illegible_biometrics / identity_conflict) that the templates alias onto —
    on dev, loosening it re-introduces false denials on truth-none packets
    (incl. MIB-000865) exactly as the prior detectors did.
* Guards: the standard two-view reader skips SAMPLE-DENIAL pages. One narrow
  exception recovers a visibly truncated token on an independently case-bound
  B-13 biometric slip when its OCR suffix has exactly one legal
  disqualifying completion and raw pixels corroborate both that completion and
  the suffix at the strip head. A value already in ``struck_values`` (colored
  vector strike) is dropped — ``decide`` re-filters it too. Office cancellation
  stamps are deliberately NOT a hard block here:
  unlike superseded fee receipts, genuine disqualifying flags print on
  ARCHIVE/COPY-stamped intake/biometric pages (dev: MIB-000397/000222/000855),
  so a stamp block would suppress true deny evidence while the two-view gate
  already yields zero false fires.

Accepted reads are injected into the risk_flags pool as
``[flag, "flag_roi", 5, score, flag]`` at harvest rank 5; because the reader
never emits "none", the injected candidate never touches the weak-"none"
approval guard, which keys on ``fields["risk_flags"] == "none"``.
"""
import os
import re

from . import ctcscore, parse_ocr, pixmatch
from .vocab import FLAGS

# label variants that anchor the flags value strip.
FLAG_LABELS = ("Observed flags:", "Flags Observed:", "Risk Flags:")

# deny-direction vocabulary: only these four ever fire.
DQ_FLAGS = ("memory_tampering", "planetary_embargo", "active_warrant",
            "biohazard_red")
# CTC ranks the read against every flag plus "none".
_CTC_CANDS = ("none",) + tuple(FLAGS)

# distinctive underscore suffix per flag (unique to that flag; long enough that
# a "none"/short-token alias cannot satisfy the suffix view).
_SUFFIX = {"memory_tampering": "_tampering", "planetary_embargo": "_embargo",
           "active_warrant": "_warrant", "biohazard_red": "ard_red"}

# Template view (both preprocessings): full word at the strip head + suffix.
FULL_NCC = 0.45          # full-word NCC floor, raw AND contrast-stretched
FULL_X_MAX = 35          # the value is left-aligned right after the label
SUF_NCC = 0.40           # distinctive-suffix NCC floor (raw)
SUF_BACK = 10            # suffix may start slightly left of the measured head
SUF_FWD = 70             # ...and up to one word-width right of it

# Glyph view (CTC): the same flag must be the recognizer's top candidate.
CTC_FLOOR = -3.5         # absolute length-normalized log-prob floor
CTC_VS_NONE = 2.0        # margin of the flag over "none" (nats)

# Partial-token exception (MIB-000855). These are deliberately stricter and
# more local than the standard template path: a damaged OCR suffix is only an
# index into the closed vocabulary; the scan must independently corroborate
# the unique full value and the suffix at the start of the anchored strip.
FRAG_LABEL_NCC = 0.60
FRAG_FULL_NCC = 0.44
FRAG_FULL_X_MAX = 15
FRAG_SUFFIX_NCC = 0.35
FRAG_SUFFIX_X_MAX = 8


def enabled():
    return os.environ.get("MIB_FLAG_ROI", "1") != "0"


def _suffix_tmpl(text):
    t = pixmatch.render_text(text, 6.7, True)   # flags style: bold 6.7pt
    return (t,) if t is not None else ()


def _crop_ink(strip):
    """Crop the value strip to its leading ink run (for the CTC recognizer)."""
    import cv2
    import numpy as np
    _, binc = cv2.threshold(strip, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col = (binc > 0).mean(axis=0)
    inked = np.where(col > 0.08)[0]
    if len(inked) == 0:
        return None
    start = int(inked[0])
    end, gap = start, 0
    for x in range(start, len(col)):
        if col[x] > 0.08:
            end, gap = x, 0
        else:
            gap += 1
            if gap > 12:
                break
    return strip[:, max(0, start - 3):min(strip.shape[1], end + 4)]


def _ctc_agrees(strip, flag):
    """True when the CTC recognizer ranks ``flag`` first among all flags+none,
    above the floor and by a wide margin over 'none'. Independent of NCC."""
    crop = _crop_ink(strip)
    if crop is None or crop.shape[1] < 8:
        return False
    try:
        scored = ctcscore.score(crop, _CTC_CANDS)
    except Exception:
        return False
    if not scored or scored[0][1] != flag:
        return False
    by_value = {v: s for s, v in scored}
    return (by_value[flag] >= CTC_FLOOR
            and by_value[flag] - by_value.get("none", -1e9) >= CTC_VS_NONE)


def _page_flag(desk, struck):
    """Return an accepted disqualifying flag for one deskewed scan page, or
    None. Requires the template view and the glyph view to agree."""
    import cv2
    anchor = pixmatch.find_label(desk, FLAG_LABELS)
    if anchor is None:
        return None
    label = anchor[5]
    strip = pixmatch._value_strip(desk, anchor)
    stretched = cv2.normalize(strip, None, 0, 255, cv2.NORM_MINMAX)
    best = None
    for flag in DQ_FLAGS:
        if flag in struck:
            continue
        full_r = pixmatch._match(strip, pixmatch._value_tmpls(label, flag))
        full_s = pixmatch._match(stretched, pixmatch._value_tmpls(label, flag))
        if not (full_r[0] >= FULL_NCC and full_s[0] >= FULL_NCC
                and full_r[1] <= FULL_X_MAX and full_s[1] <= FULL_X_MAX):
            continue
        suf = pixmatch._match(strip, _suffix_tmpl(_SUFFIX[flag]))
        if not (suf[0] >= SUF_NCC and suf[1] >= full_r[1] - SUF_BACK
                and suf[1] <= full_r[1] + SUF_FWD):
            continue
        if not _ctc_agrees(strip, flag):
            continue
        score = round(min(92.0, 70.0 + float(full_r[0]) * 30.0), 1)
        if best is None or full_r[0] > best[1]:
            best = (flag, float(full_r[0]), score)
    return best


def _case_bound_b13(case_id, lines):
    """Require one visible B-13 header and only the active case identifier."""
    if not case_id:
        return False
    joined = " ".join(str(line) for line in lines)
    upper = joined.upper()
    if "FORM B-13" not in upper or "BIOMETRIC SCAN SLIP" not in upper:
        return False
    active_match = re.fullmatch(r"MIB-(\d{6})", case_id.upper())
    if active_match is None:
        return False
    ids = set(re.findall(r"\bMIB\s*[-_:]?\s*(\d{6})\b", upper))
    return ids == {active_match.group(1)}


def _unique_dq_suffix(lines):
    """Return ``(flag, fragment)`` for one short token after a damaged
    Observed-flags label, only when the closed vocabulary has one completion."""
    values = [str(line).strip().lower() for line in lines if str(line).strip()]
    for i, line in enumerate(values):
        if not re.match(r"^obse", line):
            continue
        for raw in values[i + 1:i + 3]:
            fragment = re.sub(r"[^a-z_]", "", raw)
            if (not 5 <= len(fragment) <= 8 or "_" not in fragment
                    or not re.fullmatch(r"[a-z]+_[a-z]+", fragment)):
                continue
            matches = [flag for flag in FLAGS if flag.endswith(fragment)]
            if len(matches) == 1 and matches[0] in DQ_FLAGS:
                return matches[0], fragment
    return None


def _page_fragment_flag(desk, case_id, lines, struck):
    """Recover one uniquely completable visible suffix with pixel proof.

    This intentionally does not use the CTC recognizer: its damaged-token view
    cannot represent missing leading glyphs. Instead, the visible suffix and
    case-bound form supply one view, while four head-positioned NCC checks
    (full/suffix × raw/contrast-stretched) supply the independent pixel view.
    """
    import cv2
    if not _case_bound_b13(case_id, lines):
        return None
    unique = _unique_dq_suffix(lines)
    if unique is None:
        return None
    flag, fragment = unique
    if flag in struck:
        return None
    anchor = pixmatch.find_label(desk, FLAG_LABELS)
    if anchor is None or anchor[0] < FRAG_LABEL_NCC:
        return None
    label = anchor[5]
    strip = pixmatch._value_strip(desk, anchor)
    stretched = cv2.normalize(strip, None, 0, 255, cv2.NORM_MINMAX)
    full_r = pixmatch._match(strip, pixmatch._value_tmpls(label, flag))
    full_s = pixmatch._match(stretched, pixmatch._value_tmpls(label, flag))
    suffix_templates = _suffix_tmpl(fragment)
    suffix_r = pixmatch._match(strip, suffix_templates)
    suffix_s = pixmatch._match(stretched, suffix_templates)
    if not (full_r[0] >= FRAG_FULL_NCC
            and full_s[0] >= FRAG_FULL_NCC
            and full_r[1] <= FRAG_FULL_X_MAX
            and full_s[1] <= FRAG_FULL_X_MAX
            and suffix_r[0] >= FRAG_SUFFIX_NCC
            and suffix_s[0] >= FRAG_SUFFIX_NCC
            and suffix_r[1] <= FRAG_SUFFIX_X_MAX
            and suffix_s[1] <= FRAG_SUFFIX_X_MAX):
        return None
    score = round(min(90.0, 72.0 + float(full_r[0]) * 30.0), 1)
    return flag, float(full_r[0]), score, {
        "channel": "unique_visible_suffix",
        "fragment": fragment,
        "label_ncc": round(float(anchor[0]), 4),
        "suffix_ncc": round(float(min(suffix_r[0], suffix_s[0])), 4),
    }


def read_flags(doc, page_types_by_no, page_texts_by_no=None,
               struck_values=(), hidden_spans=None, case_id=None):
    """Read a disqualifying flag from damaged scan pages, or None.

    Returns ``(flag, score, provenance)`` for the strongest accepted page.
    The raster is read UNMASKED: hidden PDF text is never rasterized, so the
    scan pixels are injection-inert, and the two-view gate rejects any faint
    resurrection. ``page_texts_by_no`` (visible OCR lines) drives the SAMPLE-
    DENIAL watermark guard when available.
    """
    struck = {str(v).lower() for v in (struck_values or ())}
    texts = page_texts_by_no or {}
    best = None
    for page_no, img in pixmatch._p0b_scan_images(doc, None):
        page_type = page_types_by_no.get(page_no)
        if page_type == "foreign":
            continue
        provided = texts.get(page_no, [])
        joined = " ".join(t if isinstance(t, str) else str(t) for t in provided)
        watermarked = bool(joined and
                           parse_ocr.WATERMARK_RE.search(joined.upper()))
        desk, _ = pixmatch.deskew_robust(img)
        got = None if watermarked else _page_flag(desk, struck)
        meta = {"channel": "ctc_ncc"}
        # Ordinary labeled fields remain evidence even when a SAMPLE DENIAL
        # mark is present; the mark removes adjudicator-note authority, not the
        # identity of a case-bound biometric slip. This exception is limited to
        # the independently typed B-13 page and the strict suffix/pixel gate.
        if got is None and page_type == "biometric":
            fragment_got = _page_fragment_flag(
                desk, case_id, provided, struck)
            if fragment_got is not None:
                flag, full_ncc, score, meta = fragment_got
                got = (flag, full_ncc, score)
        if got is None:
            continue
        flag, full_ncc, score = got
        if best is None or full_ncc > best[1]:
            best = (flag, full_ncc, score, page_no, meta)
    if best is None:
        return None
    flag, full_ncc, score, page_no, meta = best
    provenance = {"page": page_no, "flag": flag,
                  "full_ncc": round(full_ncc, 4)}
    provenance.update(meta)
    return flag, score, provenance


def flag_roi_candidate(doc, page_types_by_no, page_texts_by_no=None,
                       struck_values=(), hidden_spans=None, case_id=None):
    """Pool candidate ``[flag, "flag_roi", 5, score, flag]`` or None."""
    if not enabled():
        return None
    read = read_flags(doc, page_types_by_no, page_texts_by_no,
                      struck_values, hidden_spans, case_id)
    if read is None:
        return None
    flag, score, _prov = read
    return [flag, "flag_roi", 5, score, flag]
