"""Independent native-pixel ledger for the two-ledger fusion.

Builds a complete extraction state from the exact native embedded scan pixels
(`forensics.native_scan_gray`, zero pixel mutation) rather than the composited
masked raster. The result is schema-compatible with the baseline state and is
decided by the same pure `mib.pipeline.decide`, so the native read is a fully
independent *result*, not a stream interleaved into the baseline.

Design invariants (see SECOND_CAMPAIGN_OUTCOME_2026-07-22.md "Clean next
architecture"):

* Selector abstention (`native_scan_gray` returns None) means "no supplement"
  for that page — never a baseline downgrade. A packet with no authorized
  native scan page yields no native ledger at all (``None``).
* Native rank-1 note authority enters only through the existing
  `_rank1_note_view` body-``Case ID:`` requirement. Notes recovered from native
  pixels that lack that binding carry *no* authority and are recorded as
  ``unbound_note_observations`` for a later denial-note census.
* Native HQ is bounded: a second (250-DPI) native pass runs only when the fast
  native read leaves a deny-relevant field unread, mirroring the baseline
  escalation rule, to cap runtime.
* Every native candidate records its exact physical provenance (page,
  native_image_sha256, dpi, rotation).
"""
from __future__ import annotations

import os

import numpy as _np

from . import extract, forensics, parse_ocr, pipeline

_DENY_ESCALATION_FIELDS = pipeline.DENY_RELEVANT + ("arrival_date",)


def _native_provenance(page_number, provenance, rotation_turns):
    return {
        "page": int(page_number),
        "native_image_sha256": provenance.get("native_image_sha256"),
        "dpi": provenance.get("output_dpi"),
        "rotation": int(rotation_turns * 90),
    }


def build_native_ledger(doc, case_id, baseline_aux=None):
    """Return a native-pixel extraction state, or ``None`` on full abstention.

    The returned dict is consumable by `mib.pipeline.decide` exactly like a
    baseline state. Enabled-only downgrade signals (identity_disqualified_pages,
    native_fallback_review_pages) are intentionally empty: an abstaining or
    quarantined page contributes nothing rather than degrading the result.

    ``baseline_aux`` (from `_extract_baseline_state`) bounds the native OCR cost
    so the double-view run stays inside the per-case deadline (see fusion
    REPORT.md section 4):

    * orientation reuse -- the native pass applies the baseline's per-page
      rotation directly instead of re-running the (multi-pass) orientation
      detector;
    * page skip -- when the baseline read every deny-relevant field, native OCR
      is skipped on pages the baseline already read cleanly (known type, no
      note); pages the baseline typed unknown or read as a note still run so a
      masked deny-trigger (e.g. MIB-000672 active_warrant, whose baseline
      risk_flags read is empty) is still recovered;
    * bounded HQ -- a native HQ pass runs only for a deny-relevant field the
      native fast read left unread AND the baseline also failed to read.
    """
    baseline_aux = baseline_aux or {}
    baseline_rot = baseline_aux.get("page_rot", {})
    baseline_missing_deny = set(baseline_aux.get("missing_deny", []))
    baseline_page_reads = baseline_aux.get("page_reads", {})

    def _skip_native_page(page_number):
        # If the baseline already read every deny-relevant field, native OCR is
        # only worth running on pages the baseline could not read cleanly or
        # that carry note authority; a masked deny-trigger keeps native active
        # for the whole case via baseline_missing_deny.
        if baseline_missing_deny:
            return False
        read = baseline_page_reads.get(page_number) \
            or baseline_page_reads.get(str(page_number))
        if not read:
            return False
        return (read.get("clean") and not read.get("note")
                and read.get("type") not in (None, "unknown"))

    visible, hidden = forensics.classify_spans(doc)

    page_texts = []
    for pno in range(len(doc)):
        page_texts.append("\n".join(s.text for s in visible if s.page == pno))

    text_fields = extract.extract_from_visible_text(case_id, page_texts)

    view_registry = pipeline._new_image_view_registry()
    per_page = []
    note_views = []
    unbound_note_observations = []
    provenance_by_field = {}
    page_provenance = {}
    ocr_quality, scan_pages = [], []
    native_pages = []
    foreign_pages = set()
    page_rot = {}
    image_views = {}
    authorized_scan_pages = 0
    view_sequence = [0]
    page_rank1_values = {}
    disqualified_pages = set()

    # Per-case native budget (mitigation c): OCR at most MIB_NATIVE_MAX_PAGES
    # scan pages, in descending expected-value order (a page the baseline could
    # not type, then a note page, then a deny-field-bearing page), and run
    # native HQ on at most MIB_NATIVE_MAX_HQ of them. This caps the worst-case
    # per-case wall on packets with many scan pages.
    max_pages = int(os.environ.get("MIB_NATIVE_MAX_PAGES", "4"))
    max_hq = int(os.environ.get("MIB_NATIVE_MAX_HQ", "2"))
    _deny_types = {"biometric", "slip", "adjudicator_note", "registry",
                   "intake", "receipt", "letter", "sponsor_letter"}

    def _page_ev(page_number):
        read = baseline_page_reads.get(page_number) \
            or baseline_page_reads.get(str(page_number)) or {}
        if read.get("note"):
            return 0
        if read.get("type") in (None, "unknown"):
            return 1
        if read.get("type") in _deny_types:
            return 2
        return 3

    scan_page_numbers = [int(p.number) for p in doc if p.get_images()]
    native_budget = set(sorted(
        scan_page_numbers, key=lambda n: (_page_ev(n), n))[:max(0, max_pages)])

    def _accepted(lines):
        skipped = (not lines and "empty") or \
            (pipeline._foreign_page(case_id, [t for t, _ in lines]) and "foreign") \
            or None
        return skipped, None if skipped else parse_ocr.parse_page(lines)

    def _record_candidate(parsed, page_number, provenance=None):
        if parsed is None:
            return
        per_page.append((view_sequence[0], int(page_number), parsed))
        view_sequence[0] += 1
        if provenance is not None:
            for field in parsed[1]:
                provenance_by_field.setdefault(field, provenance)

    def _texts(lines):
        return [line[0] if isinstance(line, (list, tuple)) else str(line)
                for line in lines]

    def _record_identity(page_number, lines):
        """Transactional per-page rank-1 identity census across fast + HQ.

        Mirrors the frozen page-isolation rule: an unsafe/foreign pass, a
        rank-1 disagreement between two physical views of the same page, or a
        cross-key policy contradiction quarantines the *whole* page — its
        candidates and note authority are all discarded. This is what stops an
        adversarial second-pass note or a benign-looking fast APPROVED from
        surviving alongside a contradicting view.
        """
        texts = _texts(lines)
        if pipeline._foreign_page_strict(case_id, texts):
            disqualified_pages.add(int(page_number))
        try:
            parsed = parse_ocr.parse_page(lines)
        except Exception:
            return
        if not pipeline._carries_rank1_authority(parsed):
            return
        tagged = pipeline._tag_rank1_view(parsed, {
            "page": int(page_number), "view": "native_census",
            "dpi": 0, "pass": "census"})
        known = page_rank1_values.setdefault(int(page_number), {})
        for field, values in pipeline._rank1_values(tagged).items():
            combined = known.setdefault(field, set()) | set(values)
            known[field] = combined
            if len(combined) > 1:
                disqualified_pages.add(int(page_number))
        if pipeline._rank1_policy_conflict(known):
            disqualified_pages.add(int(page_number))

    def _active(records):
        return [record for record in records
                if record[1] not in disqualified_pages]

    def _active_notes():
        return [view for view in note_views
                if view[2].get("_rank1_origin", {}).get("page")
                not in disqualified_pages]

    def _collect_note(parsed, lines, page_number, dpi, pass_name, source):
        """Route a native note through the exact body-ID authority gate."""
        if not pipeline._carries_rank1_authority(parsed):
            return
        alternate = pipeline._rank1_note_view(
            parsed, lines, case_id,
            {"page": int(page_number), "view": source, "dpi": int(dpi),
             "pass": pass_name})
        if alternate is not None:
            note_views.append(alternate)
        else:
            unbound_note_observations.append({
                "page": int(page_number), "dpi": int(dpi), "pass": pass_name,
                "finding": parsed[2].get("finding"),
            })

    for page in doc:
        if not page.get_images():
            native_pages.append(page)
            lines = [(line.strip(), 0.99)
                     for line in page_texts[page.number].splitlines()
                     if line.strip()]
            _record_identity(page.number, lines)
            skipped, parsed = _accepted(lines)
            # Trusted text-layer pages keep their ordinary and rank-1 authority
            # exactly as the baseline reads them (identical text).
            _record_candidate(parsed, page.number)
            continue

        kind_scan = page
        scan_pages.append(kind_scan)
        if int(page.number) not in native_budget:
            continue  # over the per-case native page budget (low expected value)
        if _skip_native_page(int(page.number)):
            continue  # baseline already read this page cleanly; native adds nothing
        img, provenance = forensics.native_scan_gray(
            doc, page, dpi=150, visible_spans=visible)
        if img is None:
            continue  # abstention: no supplement for this page
        authorized_scan_pages += 1
        # Reuse the baseline's per-page orientation instead of re-running the
        # multi-pass orientation detector (native is the same physical page).
        rot = int(baseline_rot.get(int(page.number),
                                   baseline_rot.get(str(page.number), 0)))
        if rot:
            img = _np.ascontiguousarray(_np.rot90(img, rot))
        page_rot[page.number] = rot
        lines, capture = pipeline._ocr_page_with_capture(img)
        prov = _native_provenance(page.number, provenance, rot)
        page_provenance[int(page.number)] = prov
        image_view = {
            "page": int(page.number),
            "ocr_source": "native_full_page_image",
            "native_image_sha256": provenance.get("native_image_sha256"),
            "output_width": int(provenance.get("output_width", img.shape[1])),
            "output_height": int(provenance.get("output_height", img.shape[0])),
            "output_dpi": provenance.get("output_dpi", 150),
            "ocr_retry_rotation": int(rot * 90),
        }
        image_views[page.number] = image_view
        pipeline._observe_ocr_capture(
            view_registry, capture,
            outer_rotation_degrees=int(rot * 90), image_view=image_view,
            page=int(page.number), consumer="native_ocr", pass_name="fast",
            transform="selected_ocr_input", default_source="native_full_page_image",
            default_dpi=150)
        ocr_quality.append(
            sum(conf for _, conf in lines) / len(lines) if lines else 0.0)
        _record_identity(page.number, lines)
        skipped, parsed = _accepted(lines)
        if parsed is None:
            continue
        # Ordinary fields from a native scan page are pooled without note
        # authority; adjudicator authority is granted only via the bound gate.
        _record_candidate(
            pipeline._without_rank1_authority(parsed), page.number, prov)
        _collect_note(parsed, lines, page.number, 150, "fast",
                      "native_full_page_image")

    if authorized_scan_pages == 0:
        return None

    ocr_candidates, doc_notes = parse_ocr.merge_candidates(
        [record[2] for record in _active(sorted(per_page))])
    pipeline._merge_rank1_authority(
        ocr_candidates, doc_notes, [], _active_notes())

    # Bounded native HQ: escalate only for a deny-relevant field that the
    # native fast read left unread AND the baseline also failed to read (there
    # is no point paying for a second native pass to recover a field the
    # baseline already has). Only pages the native fast pass actually read
    # (page_rot recorded) are escalated.
    missing = [field for field in _DENY_ESCALATION_FIELDS
               if field not in ocr_candidates and field not in text_fields
               and field not in doc_notes.get("absent_fields", [])
               and field in baseline_missing_deny]
    hq_pages = sorted(
        (p for p in scan_pages if int(p.number) in page_rot),
        key=lambda p: (_page_ev(int(p.number)), int(p.number)))[:max(0, max_hq)]
    hq_used = bool(missing and hq_pages)
    if hq_used:
        for page in hq_pages:
            img, provenance = forensics.native_scan_gray(
                doc, page, dpi=250, visible_spans=visible)
            if img is None:
                continue
            rot = page_rot.get(page.number, 0)
            if rot:
                img = _np.ascontiguousarray(_np.rot90(img, rot))
            lines, capture = pipeline._ocr_page_with_capture(img, hq=True)
            prov = _native_provenance(page.number, provenance, rot)
            if page.number in image_views:
                image_views[page.number]["hq_output_dpi"] = provenance.get(
                    "output_dpi", 250)
            pipeline._observe_ocr_capture(
                view_registry, capture,
                outer_rotation_degrees=int(rot * 90),
                page=int(page.number), consumer="native_ocr", pass_name="hq",
                transform="selected_ocr_input",
                default_source="native_full_page_image", default_dpi=250)
            _record_identity(page.number, lines)
            skipped, parsed = _accepted(lines)
            if parsed is None:
                continue
            _record_candidate(
                pipeline._without_rank1_authority(parsed), page.number, prov)
            _collect_note(parsed, lines, page.number, 250, "hq",
                          "native_full_page_image")

    ocr_candidates, doc_notes = parse_ocr.merge_candidates(
        [record[2] for record in _active(sorted(per_page))])
    pipeline._merge_rank1_authority(
        ocr_candidates, doc_notes, [], _active_notes())

    struck_values = sorted(forensics.struck_values(doc))
    composited_rank1_payload = pipeline._composited_rank1_attestation(
        _active_notes())

    ordered = _active(sorted(per_page))
    page_types = [record[2][0] for record in ordered]
    type_by_no = {page_number: parsed[0]
                  for _, page_number, parsed in ordered}

    pools = {}
    for field, (value, source) in text_fields.items():
        pools.setdefault(field, []).append(
            [value, source, pipeline.TEXT_SOURCE_RANK.get(source, 6), 95.0,
             value])
    for field, cands in ocr_candidates.items():
        pools.setdefault(field, []).extend([list(c) for c in cands])

    # No native pixel-decoder (pixmatch) channel here: the native OCR already
    # reads the native pixels, and a second full-resolution native_scan_gray
    # decode per page inside pixmatch.scan_images re-runs the mupdf overlay
    # inventory / footer sanitizer enough times to trip a C-extension refcount
    # crash (none_dealloc) under a full-corpus batch. The marginal fill value
    # does not justify that fragility.
    pixmatch_acceptances = []

    hidden_texts = [s.text for s in hidden]
    return {
        "case_id": case_id,
        "ledger_kind": "native",
        "pix_fired": [],
        "pools": pools,
        "doc_notes": doc_notes,
        "composited_rank1_payload": composited_rank1_payload,
        "page_types": page_types,
        "n_scan_pages": len(scan_pages),
        "authorized_scan_pages": authorized_scan_pages,
        "image_views": [image_views[p] for p in sorted(image_views)],
        "image_view_registry": pipeline._image_view_registry_snapshot(view_registry),
        "pixmatch_acceptances": pixmatch_acceptances,
        "hq_used": hq_used,
        "identity_disqualified_pages": [],
        "native_fallback_review_pages": [],
        "native_disqualified_pages": sorted(disqualified_pages),
        "unbound_note_observations": unbound_note_observations,
        "native_provenance": {
            "by_field": provenance_by_field,
            "by_page": page_provenance,
        },
        "struck_values": struck_values,
        # Container forensics are document-level (identical to the baseline's)
        # and never feed a decision or calibration feature; the native decide
        # reads its own pools, so an empty container keeps this ledger cheap.
        "container": {},
        "mean_ocr_conf": round(sum(ocr_quality) / len(ocr_quality), 2) if ocr_quality else 0.0,
        "injection": forensics.injection_signals(hidden),
        "hidden_field_mentions": pipeline._hidden_field_mentions(hidden_texts),
    }
