"""Narrow, asymmetric recovery reader for damaged Manual Adjudicator Notes.

A legible note ``Finding:`` line is already the pipeline's strongest authority
(``parse_ocr`` reads it and ``decide`` treats it as rank-1, 252/252 on dev).
This channel targets only the notes whose composited OCR left the finding
UNREAD (``doc_notes["finding"] is None``) while the note-page pixels still carry
a recoverable verdict — either in the ``Finding:`` value strip (label garbled,
value survives: e.g. MIB-000444 prints "DENIED" under an "inaing" label; MIB-
000134/000589 need pixel NCC) or in the deterministic generator ``Reason:``
sentence (e.g. "Denial supported by damaged registry evidence..." -> DENIED,
"Packet contains damaged or contradictory visible evidence." -> NEEDS_REVIEW).

Deny and review directions ONLY. APPROVED is never emitted and never enabled:
the emission vocabulary is ``{DENIED, NEEDS_REVIEW}``. An APPROVED render is
kept in the finding-value ranking purely as a REJECTION discriminator (on a
truth-APPROVED note the "DENIED" template would otherwise win the strip by
default — MIB-000176 proves it), and an approval-side ``Reason:`` sentence is a
hard abstain signal for the whole packet.

Design:

* Two independent channels, each internally two-view:
  - Finding value. ``find_label("Finding:")`` anchors the value strip; the
    strip is read by template NCC over ``{DENIED, NEEDS_REVIEW, APPROVED}`` at
    TWO preprocessings (raw + contrast-stretched). A ``{DENIED, NEEDS_REVIEW}``
    read is accepted only when both views rank the same value first, at the
    strip head, above the NCC floor and margin, and APPROVED wins neither view.
  - Reason sentence. The note page is OCR'd fast + HQ; each ``Reason:`` line is
    matched against a bank of unanimous generator reason->finding templates
    (mined corpus-wide; the finding-AGNOSTIC "Embargo home world: Wolf-1061c."
    reason is deliberately excluded because it maps to both DENIED and
    NEEDS_REVIEW). Unknown-typed pages face the stricter boundary: page-local
    case binding, at least two independent note-structure signals, and a
    DENIED reason must agree across fast+HQ or be corroborated by the direct
    finding-value channel.
* Hard guards: the finding is unread and unconflicted; the note page carries no
  recognized or fuzzy SAMPLE-DENIAL watermark; the packet names no foreign case
  id; unknown pages bind the active id locally and carry no cancellation stamp;
  no note page reads a different finding; and the value is not struck.
* Direction asymmetry is enforced by ``decide`` (see pipeline): a recovered
  DENIED may only flip NEEDS_REVIEW -> DENIED; a recovered NEEDS_REVIEW may only
  flip a terminal decision -> NEEDS_REVIEW. This module only ever RETURNS a
  finding; it never widens toward approval.
"""
import os
import re

from . import ocr, parse_ocr, pixmatch

# Emission vocabulary. APPROVED is a rejection discriminator only.
EMIT = ("DENIED", "NEEDS_REVIEW")
_BANK = ("DENIED", "NEEDS_REVIEW", "APPROVED")
_FINDING_LABEL = "Finding:"

# Finding-value NCC gates (synthetic templates on real damaged pixels read low;
# the head position + margin over the runner-up carry the discrimination). On
# dev the accepted deny/review reads cluster ncc>=0.34 at x<=11 with margin
# >=0.12; the strongest APPROVED-note false read sits off the strip head (x>=460)
# or is won by the APPROVED template.
FV_NCC_MIN = 0.33
FV_MARGIN = 0.10
FV_X_MAX = 16

# Approve-direction finding-word gate (MIB_NOTE_ROI_APPROVE, default OFF). The
# approve direction can flip a terminal decision toward APPROVED — the FA-
# adjacent direction — so it demands a much stronger positive read than
# deny/review: APPROVED must WIN both views at the strip head with a wide margin
# over BOTH DENIED and NEEDS_REVIEW, and an approve-class Reason narrative must
# independently agree. On dev the only clean APPROVED finding-word read
# (MIB-000176, ncc 0.636 margin 0.386) has no readable reason to corroborate,
# and the confirmed target MIB-000084 is sub-SNR (find_label cannot anchor the
# washed label; the value strip is noise) — so this direction fires 0 on dev and
# ships dark as explicit, human-audited insurance.
APPROVE_NCC_MIN = 0.45
APPROVE_MARGIN = 0.15
# Casing variants: the generator prints "APPROVED"; cover "Approved" too.
_APPROVE_WORDS = ("APPROVED", "Approved")

# Finding-NARRATIVE reason templates only: reasons that describe the DECISION
# itself, fuzzy-matched (rapidfuzz) to tolerate OCR garble. FACT-stating reasons
# ("Revoked sponsor: SPN-...", "Review-only risk flag present: ...") are
# deliberately excluded: a revoked sponsor can be hedged to NEEDS_REVIEW
# (MIB-000928) and a review-only flag can sit on a DENIED note (MIB-000471/466),
# so the fact does not determine the finding. Narrative reasons do — validated
# 0 direction mismatches over all 66 dev DENY/REVIEW reason lines. Approval-side
# narratives map to APPROVED and act as a hard abstain signal.
_REASON_MAP = (
    ("denial supported by damaged registry evidence", "DENIED"),
    ("packet contains damaged or contradictory visible evidence", "NEEDS_REVIEW"),
    ("clean or exception qualified packet", "APPROVED"),
    ("approval supported by surviving visible evidence", "APPROVED"),
)
REASON_FUZZ_MIN = 75          # best template score to accept
REASON_FUZZ_MARGIN = 12       # over the best DIFFERENT-finding template

# Note-structure detectors for pages the typer dropped to "unknown" (damaged
# headers). One signal is enough for the cheap OCR screen; authority requires
# at least two signals after both OCR views have run.
_NOTE_LABEL_RE = re.compile(r"(?i)^\W{0,4}(reason|finding)\W")
_FINDING_LINE_RE = re.compile(r"(?i)^\W{0,4}finding\W")
_REASON_LINE_RE = re.compile(r"(?i)^\W{0,4}reason\W")
_HEADER_FUZZ_MIN = 70
_WATERMARK_TARGET = "sampledenial"

# Office stamps that mark a note copy as superseded / non-current (reused from
# the fee lane). A superseded APPROVED note trusted over a real denial would be
# a manufactured false approval, so the approve direction hard-abstains on them;
# the deny/review asymmetry only needs them reported. INTAKE is a routing mark,
# not a cancellation, and is excluded.
_CANCEL_STAMPS = ("ARCHIVE", "FILED", "COPY", "DUPLICATE", "VOID", "SUPERSEDED",
                  "SUPERSED", "CANCELLED", "CANCELED", "DRAFT", "REISSUED",
                  "REPLACED", "EXPIRED", "SPECIMEN")


def enabled():
    return os.environ.get("MIB_NOTE_ROI", "1") != "0"


def approve_enabled():
    """Approve-direction ablation, OFF by default (the FA-adjacent flip)."""
    return os.environ.get("MIB_NOTE_ROI_APPROVE", "0") != "0"


def _reason_finding(text):
    """Fuzzy-map one Reason line to a finding (incl APPROVED) via the narrative
    reason bank, or None. Requires a clear margin over any other finding."""
    low = text.lower()
    if "reason" not in low and "eason" not in low:
        return None
    from rapidfuzz import fuzz
    norm = re.sub(r"(?i)^.*?reason\W*", "", text)
    norm = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", norm.lower())).strip()
    if len(norm) < 6:
        return None
    scored = sorted(
        ((max(fuzz.partial_ratio(k, norm), fuzz.token_set_ratio(k, norm)), f)
         for k, f in _REASON_MAP), reverse=True)
    top_score, top_finding = scored[0]
    other = max((s for s, f in scored if f != top_finding), default=0.0)
    if top_score < REASON_FUZZ_MIN or top_score - other < REASON_FUZZ_MARGIN:
        return None
    return top_finding


def _note_like(lines):
    """Whether OCR lines carry adjudicator-note structure — a Finding/Reason
    label line or a fuzzy 'Manual Adjudicator Note' header. Used to recover
    notes the page typer dropped to 'unknown' on a damaged header."""
    from rapidfuzz import fuzz
    for t in lines:
        s = t.strip()
        if _NOTE_LABEL_RE.match(s):
            return True
        if fuzz.partial_ratio("manual adjudicator note", s.lower()) >= _HEADER_FUZZ_MIN:
            return True
    return False


def _note_structure_signals(lines):
    """Positive note-form signals found in one page's visible OCR.

    Unknown pages need at least two of ``header``, ``finding``, and ``reason``;
    a lone reason sentence is prose, not enough authority to manufacture a
    rank-1 adjudicator finding.
    """
    from rapidfuzz import fuzz
    signals = set()
    for text in lines:
        line = str(text).strip()
        if _FINDING_LINE_RE.match(line):
            signals.add("finding")
        if _REASON_LINE_RE.match(line):
            signals.add("reason")
        if fuzz.partial_ratio("manual adjudicator note", line.lower()) >= _HEADER_FUZZ_MIN:
            signals.add("header")
    return signals


def _page_local_binding(case_id, lines):
    """True only when this page, not merely the packet, attests the active id."""
    texts = [str(text) for text in lines]
    compact = "".join(texts).replace(" ", "").upper()
    active = case_id.replace(" ", "").upper() in compact
    return active and not pipeline_foreign(case_id, texts)


def _watermark_suspect(lines):
    """Fail-closed OCR guard for damaged ``SAMPLE DENIAL`` watermarks.

    The exact parser regex remains the first check. A narrow fuzzy fallback
    catches observed OCR losses such as ``SAMPLE DENAL``, ``SAMPLE DEHAL``,
    ``E DENIAL``, and the all-caps collapse ``SAMRMFRENLANA`` without treating
    ordinary ``Finding: DENIED`` or denial-reason prose as a watermark.
    """
    from rapidfuzz import fuzz
    for text in lines:
        line = str(text).strip()
        if parse_ocr.WATERMARK_RE.search(line):
            return True
        compact = re.sub(r"[^a-z]", "", line.lower())
        if compact in {"sample", "sampl"}:
            return True
        if 7 <= len(compact) <= 18:
            score = fuzz.ratio(compact, _WATERMARK_TARGET)
            if score >= 72:
                return True
            if line.isupper() and compact.startswith("sam") and score >= 45:
                return True
    return False


def _cancel_stamps(text):
    """CANCEL_STAMPS present in an uppercased text blob (superseded-copy marks)."""
    upper = text.upper()
    return [s for s in _CANCEL_STAMPS if s in upper]


def _fv_rank(strip):
    """(ncc, value, x) for each bank template, best first."""
    scored = []
    for value in _BANK:
        ncc, x, _, _, _ = pixmatch._match(strip, pixmatch._syn(value, _FINDING_LABEL))
        scored.append((float(ncc), value, int(x)))
    scored.sort(reverse=True)
    return scored


def _finding_value(desk):
    """Two-preprocessing finding-value read on a deskewed note page.

    Returns an ``EMIT`` finding, the string ``"APPROVED"`` (abstain signal), or
    None. ``desk`` is a deskewed native-resolution scan page."""
    import cv2
    anchor = pixmatch.find_label(desk, (_FINDING_LABEL,),
                                 region=(0, desk.shape[0], 0, 700))
    if anchor is None:
        return None
    strip = pixmatch._value_strip(desk, anchor)
    stretched = cv2.normalize(strip, None, 0, 255, cv2.NORM_MINMAX)
    return _finding_value_from_strips((_fv_rank(strip), _fv_rank(stretched)))


def _finding_value_from_strips(views):
    """Pure two-view acceptance over pre-ranked strips (unit-testable).

    ``views`` is a pair of ranked ``[(ncc, value, x), ...]`` lists (raw,
    stretched). Returns an ``EMIT`` finding, ``"APPROVED"``, or None."""
    tops = [v[0] for v in views]
    # APPROVED wins either view at the strip head -> the note is an approval;
    # emit the abstain signal so the whole packet stands down.
    for ncc, value, x in tops:
        if value == "APPROVED" and x <= FV_X_MAX:
            return "APPROVED"
    if tops[0][1] != tops[1][1] or tops[0][1] not in EMIT:
        return None
    finding = tops[0][1]
    for ncc, value, x in tops:
        if ncc < FV_NCC_MIN or x > FV_X_MAX:
            return None
    for view in views:
        if view[0][0] - view[1][0] < FV_MARGIN:
            return None
    return finding


def _finding_value_approve(desk):
    """Strict APPROVED finding-word read for the approve ablation, or None.

    APPROVED (either casing) must win BOTH the raw and stretched views at the
    strip head, clearing APPROVE_NCC_MIN with an APPROVE_MARGIN lead over the
    best of DENIED / NEEDS_REVIEW in each view."""
    import cv2
    anchor = pixmatch.find_label(desk, (_FINDING_LABEL,),
                                 region=(0, desk.shape[0], 0, 700))
    if anchor is None:
        return None
    strip = pixmatch._value_strip(desk, anchor)
    views = (strip, cv2.normalize(strip, None, 0, 255, cv2.NORM_MINMAX))
    for view in views:
        appr = max(pixmatch._match(view, pixmatch._syn(w, _FINDING_LABEL))
                   for w in _APPROVE_WORDS)
        appr_ncc, appr_x = float(appr[0]), int(appr[1])
        deny = float(pixmatch._match(view, pixmatch._syn("DENIED", _FINDING_LABEL))[0])
        rev = float(pixmatch._match(view, pixmatch._syn("NEEDS_REVIEW", _FINDING_LABEL))[0])
        if (appr_ncc < APPROVE_NCC_MIN or appr_x > FV_X_MAX
                or appr_ncc - max(deny, rev) < APPROVE_MARGIN):
            return None
    return "APPROVED"


def _page_vote(fv, fast_reasons, hq_reasons, require_denied_corroboration=False):
    """Combine one note page's channels into (finding_or_None, approved_signal).

    ``fv`` is the finding-value read (``EMIT`` value / "APPROVED" / None).
    ``fast_reasons`` / ``hq_reasons`` are the reason-findings from each OCR pass
    (each an iterable of "DENIED"/"NEEDS_REVIEW"/"APPROVED"/None)."""
    approved = (fv == "APPROVED")
    votes = set()
    if fv in EMIT:
        votes.add(fv)
    fast = {r for r in fast_reasons if r}
    hq = {r for r in hq_reasons if r}
    if "APPROVED" in fast or "APPROVED" in hq:
        approved = True
    fast_d = fast - {"APPROVED"}
    hq_d = hq - {"APPROVED"}
    reason_options = fast_d | hq_d
    if len(reason_options) > 1:
        approved = True                    # contradictory reasons -> abstain
    elif len(reason_options) == 1:
        finding = next(iter(reason_options))
        if (finding != "DENIED" or not require_denied_corroboration
                or finding in fast_d and finding in hq_d or fv == "DENIED"):
            votes.add(finding)
    if approved or len(votes) != 1:
        return (None, approved)
    return (next(iter(votes)), approved)


def _packet_binding(case_id, page_texts_by_no):
    """(foreign_any, active_seen) over the packet's visible OCR text.

    ``foreign_any`` is True when any page confidently names a different case id
    (the frozen P0-B detector, footer-excluded, Hamming-1 tolerant).
    ``active_seen`` is True when the active id is attested anywhere in the
    packet (footer counts)."""
    foreign_any = False
    active_seen = False
    compact_active = case_id.replace(" ", "")
    for texts in page_texts_by_no.values():
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        if pipeline_foreign(case_id, texts):
            foreign_any = True
        if compact_active in " ".join(texts).replace(" ", ""):
            active_seen = True
    return foreign_any, active_seen


def pipeline_foreign(case_id, texts):
    """Frozen P0-B foreign-page detector (imported lazily to avoid a cycle)."""
    from .pipeline import _foreign_page
    return _foreign_page(case_id, texts)


def _has_correction(doc_notes):
    return bool(doc_notes.get("name_correction") or doc_notes.get("corrections"))


def _guards_ok(doc_notes):
    """Precondition guards that must hold before any recovery is attempted.

    A signed manual correction does NOT block a recovered deny/review finding:
    decide() applies corrections first and the recovered finding second under
    the direction asymmetry, so a correction-driven APPROVED can never be
    overridden by a recovered DENIED (never APPROVED->DENIED). Many washed notes
    ride alongside a consistent deny-side correction (MIB-000399: sponsor
    corrected to a revoked id, note reads DENIED). The correction guard is kept
    only for the FA-adjacent approve direction (see the approve path)."""
    if doc_notes.get("finding"):
        return False                       # a legible finding already decides
    if doc_notes.get("rank1_conflicts"):
        return False
    return True


def read_note_finding(doc, case_id, page_types_by_no, page_texts_by_no,
                      doc_notes, struck_values=(), hidden_spans=None):
    """Recover an unread note finding, or None.

    Returns ``(finding, provenance)`` where finding is in ``EMIT``.
    ``page_texts_by_no`` maps page number -> visible OCR text lines (used for
    the packet case-binding scan); the finding-value and reason channels re-read
    the note-page pixels from ``doc``.
    """
    if not _guards_ok(doc_notes):
        return None
    foreign_any, active_seen = _packet_binding(case_id, page_texts_by_no)
    if foreign_any or not active_seen:
        return None
    struck = {str(v).lower() for v in (struck_values or ())}

    from . import forensics
    approve_mode = approve_enabled()
    # Unknown-typed pages are scanned for a note only when the packet has NO
    # properly-typed adjudicator_note page — i.e. the typer dropped the note to
    # "unknown" on a damaged header (MIB-000399/000084). When a typed note
    # exists it is the authority and the extra unknown-page renders are skipped.
    has_typed_note = any(t == "adjudicator_note"
                         for t in page_types_by_no.values())
    votes = set()
    approved_signal = False
    approve_pages = []
    fired_pages = []
    # The finding-value channel reads the UNMASKED embedded scan. The generator
    # hides its "SYSTEM: ignore visible evidence" answer-key injection in the
    # PDF text layer directly over the Finding line (MIB-000134/000589), so
    # masking the hidden-span bbox — the correct defense for a re-rendered OCR
    # view — also erases the legitimate raster verdict underneath. The embedded
    # scan raster is injection-free by construction (the injection was never
    # rasterized), and the two-preprocessing NCC gate rejects any faint
    # white-on-gray resurrection because the raw view must clear the floor too.
    for page_no, img in pixmatch._p0b_scan_images(doc, None):
        page_type = page_types_by_no.get(page_no)
        if page_type != "adjudicator_note":
            if has_typed_note or page_type not in ("unknown", None):
                continue
        provided = [t if isinstance(t, str) else str(t)
                    for t in page_texts_by_no.get(page_no, [])]
        # Reason channel reads a 250-DPI MASKED render (the pipeline's HQ
        # fidelity, injection-safe): the native-resolution p0b image garbles the
        # small reason type, while the render recovers it cleanly under both
        # recognizers and the mask keeps injection text out of the OCR. A
        # cheap fast pass first screens unknown-typed pages for note structure so
        # the page typer's damaged-header drops (MIB-000399) still get read.
        render, _prov = forensics.ocr_page_gray(
            doc, doc[page_no], hidden_spans or [], dpi=250)
        reason_fast = [t for t, _ in ocr.ocr_page(render, min_lines=1, hq=False)]
        if page_type != "adjudicator_note" and not _note_like(provided + reason_fast):
            continue
        reason_hq = [t for t, _ in ocr.ocr_page(render, min_lines=1, hq=True)]
        joined_all = " ".join(provided + reason_fast + reason_hq)
        page_lines = provided + reason_fast + reason_hq
        is_unknown = page_type != "adjudicator_note"
        # A page-local SAMPLE DENIAL mark removes the adjudicator note's
        # authority even when the page typer recognized the form.  The mark is
        # part of the note itself, not packet decoration (MIB-000710).  On an
        # unknown page, a packet-level watermark observation is an additional
        # ambiguity guard; it must not suppress a different, typed note.
        if (_watermark_suspect(page_lines)
                or is_unknown and doc_notes.get("watermark_pages", 0)):
            continue                       # per-page SAMPLE-DENIAL guard
        stamps = _cancel_stamps(joined_all)
        if is_unknown:
            if not _page_local_binding(case_id, page_lines):
                continue
            if len(_note_structure_signals(page_lines)) < 2:
                continue
            if stamps:
                continue
        # Finding-value channel reads the native p0b image (the NCC bank is
        # calibrated for the 144-DPI 2x grid). deskew_robust adds an ink-
        # orientation fallback for the washed/skewed notes Hough silently punts
        # on (exactly this reader's degraded population).
        desk, _ = pixmatch.deskew_robust(img)
        fv = _finding_value(desk)
        # A typed COPY/ARCHIVE/FILED note may retain a strong direct Finding
        # pixel read (MIB-000134/000589). Stamped reason prose alone is not
        # sufficient authority.
        if stamps and fv not in EMIT:
            continue
        reason_findings = ({r for t in reason_fast for r in [_reason_finding(t)] if r}
                           | {r for t in reason_hq for r in [_reason_finding(t)] if r})
        finding, approved = _page_vote(
            fv, (_reason_finding(t) for t in reason_fast),
            (_reason_finding(t) for t in reason_hq),
            require_denied_corroboration=is_unknown)
        if approved:
            approved_signal = True
        if finding is not None:
            votes.add(finding)
            fired_pages.append((page_no, finding, fv, stamps))
        # Approve ablation (flag-gated): a strong APPROVED finding-word read AND
        # an independent approve-class Reason narrative on the same page — and
        # NEVER on a superseded-copy page (a stamped APPROVED note over a real
        # denial would be a manufactured false approval).
        if (approve_mode and not stamps and "APPROVED" in reason_findings
                and _finding_value_approve(desk) == "APPROVED"):
            approve_pages.append(page_no)

    # Deny/review recovery (default direction). An approve-class note never
    # donates a deny/review finding.
    dr_finding = None
    if not approved_signal and len(votes) == 1:
        dr_finding = next(iter(votes))
    if dr_finding is not None:
        if dr_finding.lower() in struck:
            return None
        if approve_pages:                  # a second page reads approve: conflict
            return None
        prov = {"finding": dr_finding, "pages": [p for p, _, _, _ in fired_pages],
                "channels": sorted({("fv" if fv == f else "reason")
                                    for p, f, fv, _ in fired_pages}),
                "stamps": sorted({s for _, _, _, st in fired_pages for s in st})}
        return dr_finding, prov

    # Approve recovery (ablation) — only when no deny/review finding contends and
    # no signed correction is in play (the FA-adjacent direction keeps the strict
    # correction guard: a correction could itself drive the decision).
    if (approve_mode and approve_pages and not votes
            and not _has_correction(doc_notes)):
        return "APPROVED", {"finding": "APPROVED", "pages": sorted(set(approve_pages)),
                            "channels": ["fv+reason"]}
    return None


def note_finding(doc, case_id, page_types_by_no, page_texts_by_no, doc_notes,
                 struck_values=(), hidden_spans=None):
    """Env-gated wrapper: returns a recovered finding string or None."""
    if not enabled():
        return None
    read = read_note_finding(doc, case_id, page_types_by_no, page_texts_by_no,
                             doc_notes, struck_values, hidden_spans)
    return None if read is None else read[0]
