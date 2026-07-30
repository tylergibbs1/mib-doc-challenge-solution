"""Parse OCR'd form pages into field candidates with evidence precedence.

Page templates (detected from title lines):
  adjudicator_note  Manual Adjudicator Note ("Finding: X. Reason: ...")  rank 1
  intake            FORM I-8090 Extraterrestrial Work Authorization      rank 2
  biometric         FORM B-13 Biometric Scan Slip                        rank 3
  sponsor_letter    Sponsor Attestation Letter                           rank 4
  registry          Planetary Registry Extract                           rank 5
  fee_receipt       MIB Fee Receipt                                      rank 2 (only fee source)

OCR text loses spaces and confuses characters; all matching is fuzzy and all
values snap to closed vocabularies. Watermark guard: "SAMPLE DENIAL" and office
stamps (INTAKE/ARCHIVE/FILED/COPY) are never evidence.
"""
import re

from .vocab import (CASE_RE, DATE_RE, FEES, FLAGS, PURPOSES, SPECIES,
                    SPONSOR_RE, VISAS, WORLDS, snap)

_NOSPACE = staticmethod  # placeholder to keep module flat


def _norm(line):
    return re.sub(r"[\s.]+", "", line).lower()


PAGE_TITLES = {
    "adjudicator_note": ("manualadjudicatornote", "adjudicatornote"),
    "intake": ("formi-8090", "workauthorizationintake"),
    "biometric": ("formb-13", "biometricscanslip"),
    "sponsor_letter": ("sponsorattestationletter", "attestationletter"),
    "registry": ("planetaryregistryextract", "registryextract"),
    "fee_receipt": ("mibfeereceipt", "feereceipt", "mibfee", "mibf"),
}
PAGE_RANK = {
    "adjudicator_note": 1, "intake": 2, "fee_receipt": 2, "biometric": 3,
    "sponsor_letter": 4, "registry": 5, "unknown": 6,
}

# label-normalized-prefix -> field. The tail entries are label SYNONYMS the
# public set never uses but a regenerated private layout plausibly would
# ("Fee:", "Flags Observed:", "Home Planet:", "Visa Type:") — a wrong capture
# just fails the value snap, so generic prefixes are safe.
LABELS = {
    "applicant": "applicant_name",
    "registryname": "applicant_name",
    "speciescode": "species_code",
    "speciesmatch": "species_code",
    "homeworld": "home_world",
    "homeplanet": "home_world",
    "visaclass": "visa_class",
    "visatype": "visa_class",
    "sponsorid": "sponsor_id",
    "arrivaldate": "arrival_date",
    "declaredpurpose": "declared_purpose",
    "purpose": "declared_purpose",
    "observedflags": "risk_flags",
    "riskflags": "risk_flags",
    "flagsobserved": "risk_flags",
    "feestatus": "fee_status",
}
NOISE_LINES = {"passportimage", "registryimage", "scanimage", "scantab",
               "copyartifact", "copy", "intake", "archive", "filed", "clear",
               "mib", "primaryintakerecord", "mibeyesonly"}
WATERMARK_RE = re.compile(r"(?i)s?ample\s*denia|denial\s*sample|s[a-z]*denial[a-z]*!")
_FOOT_RE = re.compile(r"(?i)packet\s*MIB[\s-]*\d{6}\s*/?\s*page")
# Deliberate-damage markers: the generator prints these where evidence was
# destroyed. They are positive proof of absence (drives NEEDS_REVIEW rules and
# stops the escalation ladder from burning seconds on unrecoverable fields).
ABSENT_RE = re.compile(r"(?i)\[?\s*[A-Z ]*?(washed\s*out|unreadable|cut\s*out|blank|redacted"
                       r"|missing|torn|whiteout|white\s*out|obscured|illegible|lost)\s*\]?\s*$")


def _mark_absent(notes, field):
    """Record an explicitly destroyed field once per parsed page."""
    if field not in notes["absent_fields"]:
        notes["absent_fields"].append(field)


# Content anchors show that the form body itself is upright. Wrapper anchors
# show only that the PDF/page overlay is upright: an embedded scan can still be
# sideways beneath an upright packet footer, stamp, watermark, or case id.
_CONTENT_ANCHORS = sorted({t for ts in PAGE_TITLES.values() for t in ts}
                          | set(LABELS))


def _page_anchor_flags(texts):
    """Return ``(content, wrapper)`` anchor observations for OCR text."""
    from rapidfuzz import fuzz
    noise = NOISE_LINES | {"packetmib"}
    content = wrapper = False
    for t in texts:
        n = _norm(t)
        if not n:
            continue
        noise_only = n in noise
        if CASE_RE.search(t.replace(" ", "")) or noise_only or WATERMARK_RE.search(t):
            wrapper = True
        # Exact office-stamp/wrapper tokens such as "INTAKE" are substrings of
        # longer form titles; do not let fuzzy title matching promote them to
        # content evidence.
        if noise_only:
            continue
        for a in _CONTENT_ANCHORS:
            if a in n or (len(a) >= 7 and len(n) >= 5 and
                          fuzz.partial_ratio(a, n) >= 82):
                content = True
                break
    return content, wrapper


def page_anchor_strength(texts):
    """Classify the strongest orientation evidence in ``texts``.

    ``content`` means a form title or field label was recognized; ``wrapper``
    means only packet-level material was recognized; ``none`` means neither.
    """
    content, wrapper = _page_anchor_flags(texts)
    return "content" if content else ("wrapper" if wrapper else "none")


def page_wrapper_anchored(texts):
    """True when packet-level wrapper material is present in ``texts``."""
    return _page_anchor_flags(texts)[1]


def page_anchored(texts):
    """Backward-compatible any-anchor predicate.

    Orientation gating should use :func:`page_anchor_strength`; callers that
    only need to know whether any packet/form landmark survived can retain this
    predicate.
    """
    return page_anchor_strength(texts) != "none"


def detect_page_type(lines):
    head = [_norm(t) for t, _ in lines[:8]]
    for ptype, titles in PAGE_TITLES.items():
        for title in titles:
            for h in head:
                if title in h or _similar(title, h):
                    return ptype
    return "unknown"


def _similar(a, b):
    from rapidfuzz import fuzz
    return fuzz.partial_ratio(a, b) >= 82 if len(b) >= 6 else False


def _fuzzy_prefix(prefix, norm):
    from rapidfuzz import fuzz
    head = norm[: len(prefix) + 2]
    return len(prefix) >= 7 and fuzz.ratio(prefix, head) >= 78


def _snap_value(field, raw):
    raw = raw.strip(" .:|")
    if not raw:
        return None, 0.0
    if field == "applicant_name":
        from rapidfuzz import fuzz
        tokens = re.findall(r"[A-Z][a-z]+", raw)
        # OCR sometimes loses the label colon, so "Applicant Nexdane Solvoss"
        # bleeds the label word into the value — strip label-like leading tokens.
        while tokens and max(fuzz.ratio(tokens[0].lower(), w)
                             for w in ("applicant", "registry", "name")) >= 70:
            tokens.pop(0)
        if len(tokens) >= 2:
            return f"{tokens[0]} {tokens[1]}", 90.0
        return None, 0.0
    if field == "sponsor_id":
        m = SPONSOR_RE.search(raw.replace(" ", "").upper().replace("O", "0").replace("SPN.", "SPN-"))
        return (m.group(), 95.0) if m else (None, 0.0)
    if field == "arrival_date":
        squeezed = raw.replace(" ", "")
        m = DATE_RE.search(squeezed)
        if m:
            # calendar-validate: OCR digit garble produces shapes like
            # "2026-03-47" that match the regex but are not dates — the
            # validator rejects them at submission time.
            from datetime import date as _date
            try:
                _date.fromisoformat(m.group())
                return m.group(), 95.0
            except ValueError:
                pass
        # OCR mangles separators ("tru202510.12" = 2025-10-12): find a year
        # digit-run and tolerate ./-/nothing between components.
        m = re.search(r"(20[23]\d)[.\-/]?(\d{1,2})[.\-/]?(\d{1,2})", squeezed)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                try:
                    from datetime import date as _date
                    return _date(y, mo, d).isoformat(), 80.0
                except ValueError:
                    pass
        return None, 0.0
    vocab = {"species_code": SPECIES, "home_world": WORLDS, "visa_class": VISAS,
             "declared_purpose": PURPOSES, "fee_status": FEES}.get(field)
    if vocab:
        value, score, _ = snap(raw, vocab, min_score=72)
        return value, score
    if field == "risk_flags":
        parts = re.split(r"[|,;]+", raw)
        snapped, failed = [], 0
        for p in parts:
            if not p.strip() or _norm(p) in {"none", "nane", "n0ne", "norne"}:
                continue
            v, s, _ = snap(p, FLAGS, min_score=78, rerank=False)
            if v:
                snapped.append(v)
            else:
                failed += 1
        if snapped:
            return "|".join(sorted(set(snapped))), 85.0
        # Content that failed to snap must NOT become a confident "none":
        # a garbled biohazard_red line reading as clean flags is the exact
        # catastrophic false-approval mechanism.
        return ("none", 40.0) if failed else ("none", 90.0)
    return None, 0.0


_NAME_VOCAB = None


def _name_lexicon():
    global _NAME_VOCAB
    if _NAME_VOCAB is None:
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "models" / "name_vocab.json"
        _NAME_VOCAB = json.loads(p.read_text()) if p.exists() else {"first": [], "last": []}
    return _NAME_VOCAB


def _bare_value(text):
    """(field, value, score) when a line that matched NO label is nevertheless
    a self-identifying legal value — the label-destroyed-value-survived damage
    pattern (bare "ED-3", "202604-09", "Barnard", "Lutari Qorul" lines measured
    in the fallback census). Closed vocabularies make this safe: the line must
    essentially BE a legal value (full-ratio, margin over the runner-up), not
    merely contain one. fee_status is deliberately excluded (paid/unpaid/waived
    words appear in receipt boilerplate and fee gates approvals)."""
    from rapidfuzz import fuzz, process

    squeezed = text.replace(" ", "")
    m = SPONSOR_RE.search(squeezed.upper().replace("O", "0").replace("SPN.", "SPN-"))
    if m:
        return "sponsor_id", m.group(), 70.0
    m = DATE_RE.search(squeezed)
    if m:
        from datetime import date as _date
        try:
            _date.fromisoformat(m.group())
        except ValueError:
            m = None                      # regex shape, not a real date
    if not m:
        m = re.search(r"(20[23]\d)[.\-/]?(\d{1,2})[.\-/]?(\d{1,2})$", squeezed)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            from datetime import date as _date
            try:
                if 1 <= mo <= 12 and 1 <= d <= 31:
                    return "arrival_date", _date(y, mo, d).isoformat(), 65.0
            except ValueError:
                pass
        m = None
    if m:
        return "arrival_date", m.group(), 70.0

    # A damaged separator can fuse a short world label to an otherwise exact,
    # self-identifying world token ("Wodd-Wolf-1061c"). Extract only complete
    # letter-hyphen-digit world shapes that match exactly one legal world; this
    # does not fuzzy-decode arbitrary prose or approve-direction values.
    world_tokens = re.findall(
        r"(?i)\b(?:home\W*)?wo[a-z]{1,6}\W+([A-Za-z]+-\d+[A-Za-z]?)",
        text)
    exact_worlds = {
        world for token in world_tokens for world in WORLDS
        if token.lower() == world.lower()
    }
    if len(exact_worlds) == 1:
        return "home_world", exact_worlds.pop(), 70.0

    toks = re.findall(r"[A-Z][a-z]+", text)
    if len(toks) == 2 and not re.search(r"\d", text):
        lex = _name_lexicon()
        ok = all(process.extractOne(t, v, scorer=fuzz.ratio)[1] >= 85
                 for t, v in zip(toks, (lex["first"], lex["last"])) if v)
        if ok and lex["first"]:
            return "applicant_name", f"{toks[0]} {toks[1]}", 68.0

    best = None
    for field, vocab in (("species_code", SPECIES), ("home_world", WORLDS),
                         ("visa_class", VISAS), ("declared_purpose", PURPOSES)):
        val, score, margin = snap(text.strip(" .:|"), vocab, min_score=85)
        if val and margin >= 8 and fuzz.ratio(str(val).lower(), _norm(text)) >= 80:
            if best:            # value legal in two fields: ambiguous, drop
                return None
            best = (field, val, 66.0)
    if best:
        return best

    parts = [p for p in re.split(r"[|,;\s]+", text) if len(p) >= 6]
    flags = [snap(p, FLAGS, min_score=85, rerank=False)[0] for p in parts]
    flags = sorted({f for f in flags if f})
    if flags and len(flags) == len(parts):
        return "risk_flags", "|".join(flags), 66.0
    return None


# $809.00 with OCR damage tolerance ($->S/5, 0->O, lost cents). A currency
# prefix or decimal suffix is mandatory and digit/letter neighbors are
# rejected, so id fragments ("MIB-100809", "SPN-8090") can never match; the
# $0.00 printed on unpaid/waived/unknown receipts never contains it.
_PAID_AMOUNT_RE = re.compile(
    r"(?<![0-9A-Za-z])(?:[$s5]\s?8[o0]9(?:[.,]\s?[o0]{2})?"
    r"|8[o0]9[.,]\s?[o0]{2})(?![0-9])",
    re.IGNORECASE)

_SUPERSEDED_RECEIPT_RE = re.compile(r"(?i)archiv|\bvoid\b|supersed")

_REVOKED_NOTE_RE = re.compile(
    r"(?i)revoked\s+sponsor\s*[:=]?\s*SPN[-\s]?([0-9OolIB]{4})")
_MINED_DIGIT_MAP = str.maketrans("OLIB", "0118")


def parse_page(lines):
    """lines: [(text, ocr_conf)]. Returns (page_type, {field: (value, score)}, notes)."""
    ptype = detect_page_type(lines)
    fields = {}
    notes = {"finding": None, "watermark": False, "stamps": [], "bio_confidence": None,
             "name_correction": None, "waiver_code": None, "absent_fields": [],
             "corrections": {}, "harvested": {}, "signed_fields": {},
             "rank1_observations": {},
             "registry_embargo": False}
    # Visible-decoy guard: a page that carries answer-key phrasing in VISIBLE
    # text gets no bare-value harvesting (labeled template reads only).
    page_norm = _norm(" ".join(t for t, _ in lines))
    harvest_ok = not any(k in page_norm for k in
                         ("answerkey", "system:", "ignorevisible"))

    texts = [t for t, _ in lines]
    for i, text in enumerate(texts):
        if WATERMARK_RE.search(text):
            notes["watermark"] = True
            continue
        norm = _norm(text)
        if norm in NOISE_LINES:
            continue

        m = re.search(r"(?i)biometric\s*conf\w*[:.\s]*(\d{1,3})\s*%", text)
        if m and ptype == "biometric":
            notes["bio_confidence"] = int(m.group(1))

        # Signed manual corrections are rank-1 evidence (adjudicator note
        # class). OCR routinely fuses the spaces ("Manualcorrection:
        # sponsorisSPN-4705.", "applicantisOridaneSoltari."), and the corpus
        # census found four corrected fields: applicant, sponsor, visa class,
        # fee status. \s* tolerates both spaced and fused forms.
        m = re.search(r"(?i)manual\s*cor+ections?[:.\s]*"
                      r"(applicant|sponsor|visa\s*class|fee\s*status)\s*is\s*"
                      r"([A-Za-z0-9 /-]+)", text)
        if m:
            key = re.sub(r"\s", "", m.group(1).lower())
            raw_val = m.group(2).strip(" .")
            if key == "applicant":
                nm = re.match(r"([A-Z][a-z]+)\s*([A-Z][a-z]+)$", raw_val)
                if nm:
                    value = f"{nm.group(1)} {nm.group(2)}"
                    notes["name_correction"] = value
                    observed = notes["rank1_observations"].setdefault(
                        "applicant_name", [])
                    if value not in observed:
                        observed.append(value)
            elif key == "sponsor":
                sm = SPONSOR_RE.search(raw_val.replace(" ", "").upper()
                                       .replace("O", "0").replace("SPN.", "SPN-"))
                if sm:
                    value = sm.group()
                    notes.setdefault("corrections", {})["sponsor_id"] = value
                    observed = notes["rank1_observations"].setdefault(
                        "sponsor_id", [])
                    if value not in observed:
                        observed.append(value)
            elif key == "visaclass":
                v, _, _ = snap(raw_val.upper(), VISAS, min_score=70)
                if v:
                    notes.setdefault("corrections", {})["visa_class"] = v
                    observed = notes["rank1_observations"].setdefault(
                        "visa_class", [])
                    if v not in observed:
                        observed.append(v)
            elif key == "feestatus":
                v, _, _ = snap(raw_val.lower(), FEES, min_score=75)
                if v:
                    notes.setdefault("corrections", {})["fee_status"] = v
                    observed = notes["rank1_observations"].setdefault(
                        "fee_status", [])
                    if v not in observed:
                        observed.append(v)

        # Registry pages print "Registry Status: EMBARGO REVIEW" (vs CLEAR).
        # Never APPROVED in any labeled case (0/22 on dev; 20 DENIED / 2 NR),
        # so it is used strictly as an approval blocker — CLEAR is never
        # evidence of anything (18 truth-DENIED training cases print CLEAR).
        # OCR fragments the phrase ("EMB" / "BARGOREVIEW"), so the status line
        # and the next two lines are checked for embargo-shaped heads.
        if ptype == "registry" and not notes["registry_embargo"]:
            if re.search(r"(?i)embargo\s*rev|mbargorev|bargorev", text):
                notes["registry_embargo"] = True
            elif re.search(r"(?i)registry\s*status", text):
                for j in (i + 1, i + 2):
                    if j < len(texts) and re.match(r"(?i)\W*(emb|bargo|rgorev)", texts[j]):
                        notes["registry_embargo"] = True
                        break

        m = re.search(r"(?i)waiver\s*code[:.\s]*([A-Z0-9/-]+)", text)
        if m and ptype == "fee_receipt":
            notes["waiver_code"] = m.group(1)
        elif ptype == "fee_receipt" and _norm(text) == "waivercode":
            nxt = texts[i + 1].strip() if i + 1 < len(texts) else ""
            if nxt:
                notes["waiver_code"] = nxt

        # A signed note's "Reason:" line states the deciding field value in
        # clean prose ("Reason: Disqualifying risk flag: biohazard_red.",
        # "Reason: Mandatory fee unpaid.") — rank-1 evidence the field pools
        # were never fed (the census found ~25 dev cases losing exactly these
        # extraction points while the Finding already drove the decision).
        # Deny/review-side values only, by construction: notes never print
        # "fee paid" or "flags none" reasons, so the harvest cannot move a
        # case toward approval.
        if (ptype == "adjudicator_note" and not notes["watermark"]
                and not WATERMARK_RE.search(text)):
            m = re.search(r"(?i)re[ao]son\W*(?:disqualifying|review[\s-]*only)?"
                          r"\W*(?:risk|nsk)\s*fl?ags?\W*(?:presen\w*)?\W*[:.]?\s*"
                          r"([a-z_ |,]+)", text)
            if m:
                parts = re.split(r"[|,;\s]+", m.group(1))
                got = sorted({v for p in parts if len(p) >= 6
                              for v, _, _ in [snap(p, FLAGS, min_score=82, rerank=False)]
                              if v})
                if got:
                    prev = fields.get("risk_flags")
                    merged = sorted(set(got) |
                                    (set(prev[0].split("|")) - {"none"} if prev else set()))
                    candidate = ("|".join(merged), 96.0, m.group(1).strip())
                    fields["risk_flags"] = candidate
                    signed = notes["signed_fields"].setdefault("risk_flags", [])
                    if candidate[0] not in {item[0] for item in signed}:
                        signed.append(candidate)
            m = re.search(r"(?i)reason\W*(?:mandatory\s*)?fee\s*"
                          r"(?:status\s*)?(unknown|unpaid)", text)
            if m:
                candidate = (m.group(1).lower(), 96.0, text.strip()[:40])
                fields["fee_status"] = candidate
                signed = notes["signed_fields"].setdefault("fee_status", [])
                if candidate[0] not in {item[0] for item in signed}:
                    signed.append(candidate)
            # Note reasons name revoked sponsors verbatim ("Reason: Revoked
            # sponsor: SPN-2718.") — mined for batch-level rotation
            # insurance; acceptance thresholds live at the batch layer.
            m = _REVOKED_NOTE_RE.search(text)
            if m:
                digits = m.group(1).upper().translate(_MINED_DIGIT_MAP)
                if digits.isdigit():
                    notes.setdefault("mined_revoked", []).append(
                        "SPN-" + digits)

        m = re.search(r"(?i)finding[:.\s]*(approved|denied|needs[\s_-]*review|[A-Z_]{6,})", text)
        if m and ptype == "adjudicator_note" and not WATERMARK_RE.search(text):
            verdict, _, _ = snap(re.sub(r"[\s-]+", "_", m.group(1).upper()),
                                 ["APPROVED", "DENIED", "NEEDS_REVIEW"], min_score=70)
            notes["finding"] = verdict
            if verdict:
                observed = notes["rank1_observations"].setdefault(
                    "finding", [])
                if verdict not in observed:
                    observed.append(verdict)
        # Damage sometimes eats the "Finding:" label itself ("g:NEEDS_REVIEW").
        # On a note page a standalone uppercase verdict line — at most a tiny
        # damaged prefix before it — is still the finding. Strict by design:
        # whole-line match only, uppercase only, never on watermarked pages
        # (a "SAMPLE DENIAL" page must not donate a bare DENIED).
        if (ptype == "adjudicator_note" and not notes["watermark"]
                and not WATERMARK_RE.search(text)):
            m = re.match(r"^[^A-Za-z]{0,3}(?:[a-z]{1,5}[:.])?\s*"
                         r"(APPROVED|DENIED|NEEDS[\s_-]*REVIEW)\W*$", text.strip())
            if m:
                verdict, _, _ = snap(re.sub(r"[\s-]+", "_", m.group(1)),
                                     ["APPROVED", "DENIED", "NEEDS_REVIEW"], min_score=70)
                if notes["finding"] is None:
                    notes["finding"] = verdict
                if verdict:
                    observed = notes["rank1_observations"].setdefault(
                        "finding", [])
                    if verdict not in observed:
                        observed.append(verdict)

        # Same-line "Label: value" — exact prefix first, then fuzzy prefix for
        # OCR-garbled labels ("Obserdfags:none").
        matched = False
        for prefix, field in LABELS.items():
            exact = norm.startswith(prefix) and len(norm) > len(prefix) + 1
            fuzzy = (not exact and len(norm) > len(prefix) and
                     _fuzzy_prefix(prefix, norm))
            if exact or fuzzy:
                parts = re.split(r"[:.]", text, 1)
                raw_seg = parts[-1]
                separator_label = _norm(parts[0]) if len(parts) == 2 else ""
                separator_follows_label = (
                    (separator_label.startswith(prefix)
                     and len(separator_label) <= len(prefix) + 2)
                    or (abs(len(separator_label) - len(prefix)) <= 2
                        and _fuzzy_prefix(prefix, separator_label))
                )
                if (separator_follows_label
                        and ABSENT_RE.fullmatch(raw_seg.strip())):
                    _mark_absent(notes, field)
                    matched = True
                    break
                value, score = _snap_value(field, raw_seg)
                if value and (field not in fields or score > fields[field][1]):
                    fields[field] = (value, score if exact else score - 5, raw_seg.strip())
                matched = True
                break
        if matched:
            continue

        # Truncated/garbled label before a separator ("licant:Xanul Lunax",
        # "eciesCode:ALPHA_DRACONIAN", "Vlsa Closs:MED-3"): damage eats the
        # label's head, the value survives after the colon. Match the
        # pre-separator segment against the label set as a whole.
        m = re.match(r"([^:]{4,24}):(.+)$", text)
        if m:
            pre = _norm(m.group(1))
            if len(pre) >= 4 and pre not in NOISE_LINES:
                from rapidfuzz import fuzz as _f
                best_field, best_sc = None, 0.0
                for prefix, field in LABELS.items():
                    sc = max(_f.partial_ratio(prefix, pre), _f.ratio(prefix, pre))
                    if sc > best_sc:
                        best_field, best_sc = field, sc
                if best_field and best_sc >= 80:
                    if ABSENT_RE.fullmatch(m.group(2).strip()):
                        _mark_absent(notes, best_field)
                        continue
                    value, score = _snap_value(best_field, m.group(2))
                    if value and (best_field not in fields or score - 8 > fields[best_field][1]):
                        fields[best_field] = (value, score - 8, m.group(2).strip())
                    continue

        # Label-only line: value on the following line (or preceding, for the
        # intake form's occasional value-before-label ordering).
        label_only = False
        for prefix, field in LABELS.items():
            if norm == prefix or (len(norm) > 3 and _similar(prefix, norm) and abs(len(norm) - len(prefix)) <= 2):
                label_only = True
                for j in (i + 1, i - 1):
                    if 0 <= j < len(texts) and _norm(texts[j]) not in LABELS and _norm(texts[j]) not in NOISE_LINES:
                        if ABSENT_RE.fullmatch(texts[j].strip()):
                            _mark_absent(notes, field)
                            break
                        value, score = _snap_value(field, texts[j])
                        if value:
                            if field not in fields or score > fields[field][1]:
                                fields[field] = (value, score, texts[j].strip())
                            break
                break
        if label_only:
            continue

        # Nothing matched: the label may be destroyed while the value survived.
        if harvest_ok and not _FOOT_RE.search(text):
            hv = _bare_value(text)
            if hv and hv[0] not in notes["harvested"]:
                notes["harvested"][hv[0]] = (hv[1], hv[2], text.strip())

    # Fee-amount paid indicator: the generator prints the $809.00 charge only
    # on paid receipts (unpaid/waived/unknown receipts print $0.00 — 118/118
    # paid vs 0/59 others on the text-layer census), so a legible amount is
    # decisive visible evidence of `paid` even when the status word itself is
    # damaged. Same-page fallback only (a direct status read always wins),
    # never on superseded receipts (ARCHIVE trap class), and never when the
    # document explicitly marks the status as absent/illegible — those cases
    # are under-determined by design and must keep hedging.
    if (ptype == "fee_receipt" and "fee_status" not in fields
            and "fee_status" not in notes["absent_fields"]
            and not any(_SUPERSEDED_RECEIPT_RE.search(t) for t in texts)
            and any(_PAID_AMOUNT_RE.search(t) for t in texts)):
        fields["fee_status"] = ("paid", 90.0, "amount_809_paid_indicator")
    return ptype, fields, notes


def merge_candidates(per_page):
    """per_page: [(page_type, fields, notes)].

    Returns (candidates, doc_notes) where candidates maps
    field -> list of (value, source, rank, score) across all pages.
    """
    candidates = {}
    doc_notes = {"finding": None, "finding_rank": 99, "watermark_pages": 0,
                 "bio_confidence": None, "name_correction": None, "waiver_code": None,
                 "absent_fields": [], "corrections": {}, "registry_embargo": False}
    for ptype, fields, notes in per_page:
        rank = PAGE_RANK.get(ptype, 6)
        if notes["watermark"]:
            doc_notes["watermark_pages"] += 1
        if notes.get("bio_confidence") is not None:
            doc_notes["bio_confidence"] = notes["bio_confidence"]
        if notes.get("name_correction"):
            doc_notes["name_correction"] = notes["name_correction"]
        for f, v in notes.get("corrections", {}).items():
            doc_notes["corrections"].setdefault(f, v)
        if notes.get("waiver_code"):
            doc_notes["waiver_code"] = notes["waiver_code"]
        for sponsor in notes.get("mined_revoked", []):
            doc_notes.setdefault("mined_revoked", []).append(sponsor)
        if notes.get("registry_embargo"):
            doc_notes["registry_embargo"] = True
        for f in notes.get("absent_fields", []):
            if f not in doc_notes["absent_fields"]:
                doc_notes["absent_fields"].append(f)
        if notes["finding"] and rank < doc_notes["finding_rank"]:
            doc_notes["finding"] = notes["finding"]
            doc_notes["finding_rank"] = rank
        for field, (value, score, raw) in fields.items():
            candidates.setdefault(field, []).append((value, ptype, rank, score, raw))
        # Harvested bare values enter the pool at the LOWEST precedence: any
        # labeled read of the field, on any page, outranks them.
        for field, (value, score, raw) in notes.get("harvested", {}).items():
            candidates.setdefault(field, []).append((value, f"{ptype}_bare", 6, score, raw))
    return candidates, doc_notes
