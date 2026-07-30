"""Narrow embargo-world reader for damaged scan pages (deny-direction only).

The pixel decoder (``mib/pixmatch.py``) reads ``home_world`` with a whole-string
NCC gate (ncc 0.55 / margin 0.12, 96% at n=57 on the eval half) and abstains
when the margin collapses — which is exactly what happens on damaged 7pt world
values, where "Wolf-1061c" and "Proxima-b" sit within 0.02 NCC of each other
(MIB-000261 p2 prints a clean "Wolf-1061c" that the channel ranks second by
0.016, far under the 0.12 gate). Lowering that gate is out of scope; instead
this channel adds a SECOND, glyph-aware read for the embargo worlds only.

Embargo-direction ONLY: it emits a world only from
``rules.HARD_EMBARGO_WORLDS | rules.SOFT_EMBARGO_WORLDS`` (Eris Relay,
TRAPPIST-1e, Wolf-1061c). A hard-embargo world always blocks approval and a
soft-embargo world blocks it for non-DIP-1 visas (``decide`` applies the visa
exemption downstream), so a fire can only move a case toward denial — never
enable an approval a clean read would not have.

Design (validated on dev, 53 world-unread packets; census 1/1 fire truth-
correct, 0 wrong, recovers the human seed MIB-000261 organically):

* Anchored ROI. ``find_label("Home World:")`` anchors the value strip.
* TWO INDEPENDENT VIEWS must agree on the SAME embargo world:
  - Glyph recognition. The shipped CTC recognizer (``mib/ctcscore.py``) must
    rank an embargo world first among ALL worlds, above an absolute log-prob
    floor and by a wide margin over the runner-up. This is the discrimination
    whole-string NCC lacks at 7pt; every false embargo candidate on dev sits
    below the floor or under the margin.
  - Template correlation. That same world's whole-string NCC (bank + synthetic)
    must clear a floor at the strip head under both raw and contrast-stretched
    preprocessings, so the glyph read is corroborated by the pixels and not a
    recognizer hallucination on a blank strip.
* Guards: watermarked (SAMPLE-DENIAL) and foreign pages are skipped; a struck
  value is dropped.

Accepted reads are injected into the home_world pool as
``[world, "world_roi", 5, score, world]`` at harvest rank 5, only when the pool
is otherwise empty; ``decide`` applies embargo adjudication and the DIP-1
exemption unchanged.
"""
import os

from . import ctcscore, parse_ocr, pixmatch, rules
from .vocab import WORLDS

_WORLD_LABEL = "Home World:"
EMBARGO_WORLDS = frozenset(rules.HARD_EMBARGO_WORLDS | rules.SOFT_EMBARGO_WORLDS)

# Glyph view (CTC): an embargo world must be the recognizer's top world.
CTC_FLOOR = -3.0         # absolute length-normalized log-prob floor
CTC_MARGIN = 2.0         # margin over the runner-up world (nats)
# Template view (NCC): the same world corroborated at the strip head.
NCC_MIN = 0.50           # whole-string NCC floor, raw AND contrast-stretched
NCC_X_MAX = 20           # left-aligned right after the label


def enabled():
    return os.environ.get("MIB_WORLD_ROI", "1") != "0"


def _crop_ink(strip):
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
            if gap > 14:
                break
    return strip[:, max(0, start - 3):min(strip.shape[1], end + 4)]


def _ctc_embargo(strip):
    """Return an embargo world the CTC recognizer ranks first among all worlds,
    above the floor and by a wide margin, or None. Independent of NCC."""
    crop = _crop_ink(strip)
    if crop is None or crop.shape[1] < 8:
        return None
    try:
        scored = ctcscore.score(crop, WORLDS)
    except Exception:
        return None
    if len(scored) < 2:
        return None
    (top_s, top_w), (second_s, _) = scored[0], scored[1]
    if top_w not in EMBARGO_WORLDS:
        return None
    if top_s < CTC_FLOOR or top_s - second_s < CTC_MARGIN:
        return None
    return top_w


def _page_world(desk, struck):
    """Return an accepted embargo world for one deskewed scan page, or None.
    Requires the glyph view and the template view to agree."""
    import cv2
    anchor = pixmatch.find_label(desk, (_WORLD_LABEL,))
    if anchor is None:
        return None
    strip = pixmatch._value_strip(desk, anchor)
    world = _ctc_embargo(strip)
    if world is None or world.lower() in struck:
        return None
    stretched = cv2.normalize(strip, None, 0, 255, cv2.NORM_MINMAX)
    tmpls = pixmatch._value_tmpls(_WORLD_LABEL, world)
    raw = pixmatch._match(strip, tmpls)
    stc = pixmatch._match(stretched, tmpls)
    if not (raw[0] >= NCC_MIN and stc[0] >= NCC_MIN
            and raw[1] <= NCC_X_MAX and stc[1] <= NCC_X_MAX):
        return None
    score = round(min(92.0, 70.0 + float(raw[0]) * 30.0), 1)
    return world, float(raw[0]), score


def read_world(doc, page_types_by_no, page_texts_by_no=None,
               struck_values=(), hidden_spans=None):
    """Read an embargo home world from damaged scan pages, or None.

    Returns ``(world, score, provenance)`` for the strongest accepted page.
    The raster is read UNMASKED (injection-inert by construction);
    ``page_texts_by_no`` drives the SAMPLE-DENIAL watermark guard.
    """
    struck = {str(v).lower() for v in (struck_values or ())}
    texts = page_texts_by_no or {}
    best = None
    for page_no, img in pixmatch._p0b_scan_images(doc, None):
        if page_types_by_no.get(page_no) == "foreign":
            continue
        provided = texts.get(page_no, [])
        joined = " ".join(t if isinstance(t, str) else str(t) for t in provided)
        if joined and parse_ocr.WATERMARK_RE.search(joined.upper()):
            continue
        desk, _ = pixmatch.deskew_robust(img)
        got = _page_world(desk, struck)
        if got is None:
            continue
        world, ncc, score = got
        if best is None or ncc > best[1]:
            best = (world, ncc, score, page_no)
    if best is None:
        return None
    world, ncc, score, page_no = best
    return world, score, {"page": page_no, "world": world,
                          "ncc": round(ncc, 4)}


def world_roi_candidate(doc, page_types_by_no, page_texts_by_no=None,
                        struck_values=(), hidden_spans=None):
    """Pool candidate ``[world, "world_roi", 5, score, world]`` or None."""
    if not enabled():
        return None
    read = read_world(doc, page_types_by_no, page_texts_by_no,
                      struck_values, hidden_spans)
    if read is None:
        return None
    world, score, _prov = read
    return [world, "world_roi", 5, score, world]
