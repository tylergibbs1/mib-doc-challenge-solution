"""Deny-direction recovery for a damaged, visibly revoked sponsor ID.

This is intentionally not a general sponsor decoder.  A fire is possible only
when two independently rendered OCR views agree on the same four trailing
digits, those digits complete one sponsor explicitly present in the policy's
revoked set, the page binds the active case, and the scan independently
contains strong ``Sponsor ID:`` and ``SPN-`` pixel anchors.  Consequently the
channel can only add a revoked sponsor and move a packet toward denial; it
cannot supply benign sponsor evidence or enable approval.
"""
import os
import re

from . import pixmatch, rules

LABEL_NCC = 0.70
PREFIX_NCC = 0.65
_DAMAGED_TOKEN_RE = re.compile(r"\b[12]\d{3}[-‐‑–—](\d{4})\b")
_CASE_RE = re.compile(r"\bMIB\s*[-_:]?\s*(\d{6})\b", re.I)


def enabled():
    return os.environ.get("MIB_REVOKED_SPONSOR_ROI", "1") != "0"


def _page_bound(case_id, *line_groups):
    match = re.fullmatch(r"MIB-(\d{6})", str(case_id).upper())
    if match is None:
        return False
    joined = " ".join(
        str(line) for lines in line_groups for line in (lines or ()))
    ids = set(_CASE_RE.findall(joined))
    return ids == {match.group(1)}


def _revoked_suffix(lines):
    """The one revoked sponsor implied by damaged ``20xx-####`` OCR."""
    suffixes = {
        match.group(1)
        for line in (lines or ())
        for match in _DAMAGED_TOKEN_RE.finditer(str(line))
    }
    revoked = {
        f"SPN-{suffix}" for suffix in suffixes
        if f"SPN-{suffix}" in rules.REVOKED_SPONSORS
    }
    return next(iter(revoked)) if len(revoked) == 1 else None


def read_revoked_sponsor(doc, case_id, page_types_by_no,
                         fast_lines_by_page, hq_lines_by_page,
                         struck_values=(), hidden_spans=None):
    """Return ``(sponsor, score, provenance)`` or ``None``.

    Fast and HQ OCR must independently imply the same revoked sponsor.  Pixel
    checks use the hidden-span-masked P0-B scan view and corroborate only the
    field label and ``SPN-`` prefix; the OCR views supply the agreeing digits.
    """
    struck = {str(value).upper() for value in (struck_values or ())}
    fast = fast_lines_by_page or {}
    hq = hq_lines_by_page or {}
    for page_no, image in pixmatch._p0b_scan_images(doc, hidden_spans):
        page_type = page_types_by_no.get(page_no)
        if page_type not in ("intake", "registry", "unknown", None):
            continue
        fast_lines = fast.get(page_no, ())
        hq_lines = hq.get(page_no, ())
        if not fast_lines or not hq_lines:
            continue
        if not _page_bound(case_id, fast_lines, hq_lines):
            continue
        fast_value = _revoked_suffix(fast_lines)
        hq_value = _revoked_suffix(hq_lines)
        if fast_value is None or fast_value != hq_value or fast_value in struck:
            continue
        desk, _ = pixmatch.deskew_robust(image)
        anchor = pixmatch.find_label(
            desk, pixmatch.FIELD_LABELS["sponsor_id"])
        if anchor is None or anchor[0] < LABEL_NCC:
            continue
        decoded = pixmatch.decode_field(desk, "sponsor_id")
        if decoded is None or decoded.get("prefix_ncc", 0.0) < PREFIX_NCC:
            continue
        score = round(min(92.0, 70.0 + float(anchor[0]) * 25.0), 1)
        return fast_value, score, {
            "page": page_no,
            "sponsor": fast_value,
            "channel": "two_ocr_suffix_plus_prefix",
            "label_ncc": round(float(anchor[0]), 4),
            "prefix_ncc": round(float(decoded["prefix_ncc"]), 4),
        }
    return None


def revoked_sponsor_candidate(doc, case_id, page_types_by_no,
                              fast_lines_by_page, hq_lines_by_page,
                              struck_values=(), hidden_spans=None):
    """Pool candidate ``[value, source, rank, score, raw]`` or ``None``."""
    if not enabled():
        return None
    read = read_revoked_sponsor(
        doc, case_id, page_types_by_no, fast_lines_by_page, hq_lines_by_page,
        struck_values, hidden_spans)
    if read is None:
        return None
    value, score, _provenance = read
    return [value, "revoked_sponsor_roi", 5, score, value]
