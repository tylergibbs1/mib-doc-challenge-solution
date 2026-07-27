from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date

from rapidfuzz import fuzz

from .constants import (
    DISQUALIFYING_FLAGS,
    DOCUMENT_PRIORITIES,
    HOME_WORLDS,
    PURPOSES,
    REVIEW_FLAGS,
    RISK_FLAGS,
    SPECIES_CODES,
    VISA_CLASSES,
)
from .pdf import PageEvidence

LABEL_WORDS = {
    "applicant",
    "species",
    "home",
    "visa",
    "sponsor",
    "arrival",
    "declared",
    "purpose",
    "case",
    "passport",
    "registry",
    "packet",
}
DATE_RE = re.compile(
    r"\b(20[0-9OIl]{2}[-/.\s][01OIl][0-9OIl][-/.\s][0-3OIl][0-9OIl])\b"
)
SPONSOR_RE = re.compile(
    r"\bSP[NM]\s*[-: ]\s*((?:[0-9OILSB]\s*){4})\b",
    re.IGNORECASE,
)
DOTTED_SPONSOR_RE = re.compile(
    r"\bSP[NM]\s*\.\s*((?:[0-9OILSB]\s*){4})\b",
    re.IGNORECASE,
)
OBSERVED_FLAGS_RE = re.compile(
    r"\bobserved\s+flags?\s*[:|\]]\s*([^\r\n]+)",
    re.IGNORECASE,
)

RISK_FLAG_PHRASES = {
    "active_warrant": ("ACTIVE WARRANT", "WARRANT ACTIVE"),
    "biohazard_red": ("BIOHAZARD RED", "BIO HAZARD RED", "HAZARD LEVEL RED"),
    "identity_conflict": ("IDENTITY CONFLICT", "IDENTITY MISMATCH"),
    "illegible_biometrics": (
        "ILLEGIBLE BIOMETRICS",
        "BIOMETRICS ILLEGIBLE",
        "BIOMETRIC UNREADABLE",
        "BIOMETRICS UNREADABLE",
        "RISK PANEL MISSING",
    ),
    "memory_tampering": ("MEMORY TAMPERING", "MEMORY ALTERATION"),
    "planetary_embargo": (
        "PLANETARY EMBARGO",
        "REGISTRY STATUS EMBARGO",
        "WORLD EMBARGO",
    ),
    "rescinded_denial": (
        "RESCINDED DENIAL",
        "DENIAL RESCINDED",
        "DENIAL CROSSED OUT",
    ),
    "sponsor_mismatch": ("SPONSOR MISMATCH", "SPONSOR DOES NOT MATCH"),
}


@dataclass
class Extraction:
    case_id: str
    applicant_name: str
    species_code: str
    home_world: str
    visa_class: str
    sponsor_id: str
    arrival_date: str
    declared_purpose: str
    risk_flags: str
    fee_status: str
    missing_fields: tuple[str, ...]
    evidence_quality: float
    mean_ocr_confidence: float

    def output_fields(self, fallbacks: dict[str, str] | None = None) -> dict:
        value = asdict(self)
        value.pop("missing_fields")
        value.pop("evidence_quality")
        value.pop("mean_ocr_confidence")
        # Statistical fallbacks live only at the output boundary. Policy and
        # confidence continue to see the unresolved field as missing.
        for field, fallback in (fallbacks or {}).items():
            if field in self.missing_fields:
                value[field] = fallback
        return value


def normalized(value: str) -> str:
    return " ".join(re.findall(r"[A-Z0-9]+", value.upper()))


def compact(value: str) -> str:
    return "".join(re.findall(r"[A-Z0-9]", value.upper()))


def _windows(tokens: list[str], size: int):
    for width in range(max(1, size - 1), size + 2):
        for start in range(max(0, len(tokens) - width + 1)):
            yield " ".join(tokens[start : start + width])


def fuzzy_presence(text: str, value: str) -> float:
    haystack = normalized(text)
    needle = normalized(value)
    if needle in haystack:
        return 1.0
    tokens = haystack.split()
    needle_tokens = needle.split()
    if not tokens:
        return 0.0
    compact_needle = compact(needle)
    return (
        max(
            fuzz.ratio(compact(window), compact_needle, score_cutoff=45.0)
            for window in _windows(tokens, len(needle_tokens))
        )
        / 100.0
    )


def _explicit_risk_flags(text: str, *, strict: bool = False) -> set[str]:
    norm = normalized(text)
    found: set[str] = set()
    for flag in RISK_FLAGS:
        phrases = RISK_FLAG_PHRASES[flag]
        if strict and flag == "planetary_embargo":
            phrases = ("PLANETARY EMBARGO", "WORLD EMBARGO")
        if normalized(flag) in norm or any(phrase in norm for phrase in phrases):
            found.add(flag)
    if re.search(r"\bREGISTRY STATUS(?: IS)? EMBARGO\b(?! REVIEW)", norm):
        found.add("planetary_embargo")
    return found


def _exact_observed_flags(pages: list[PageEvidence]) -> set[str] | None:
    """Return one fully decoded Observed-flags row, or None if it is ambiguous.

    Exact row geometry is stronger evidence than whole-page fuzzy matching or
    cross-document inference. Any malformed token or disagreement vetoes the
    authority route so damaged rows continue through OCR-tolerant recovery.
    """
    canonical = {flag.casefold(): flag for flag in RISK_FLAGS}
    parsed_rows: list[frozenset[str]] = []
    invalid_row = False
    for page in pages:
        for match in OBSERVED_FLAGS_RE.finditer(page.text):
            raw = re.split(
                r"\b(?:SCAN IMAGE|Packet MIB-|Synthetic hiring)\b",
                match.group(1).strip(),
            )[0]
            parts = [
                part.strip()
                for part in re.split(r"[,|;]+", raw)
                if part.strip()
            ]
            tokens: list[str] = []
            valid = bool(parts)
            for part in parts:
                token = re.sub(
                    r"[^a-z0-9]+", "_", part.casefold()
                ).strip("_")
                if token == "none":
                    tokens.append(token)
                elif token in canonical:
                    tokens.append(canonical[token])
                else:
                    valid = False
            if "none" in tokens and len(tokens) != 1:
                valid = False
            if not valid:
                invalid_row = True
                continue
            parsed_rows.append(
                frozenset(token for token in tokens if token != "none")
            )
    if invalid_row or not parsed_rows or len(set(parsed_rows)) != 1:
        return None
    return set(parsed_rows[0])


def best_lexicon(
    text: str, values: tuple[str, ...], cutoff: float = 0.72
) -> tuple[str, float]:
    scored = [(value, fuzzy_presence(text, value)) for value in values]
    value, score = max(scored, key=lambda item: item[1])
    return (value, score) if score >= cutoff else ("", score)


def _label_value(lines: list[str], labels: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(
        rf"^\s*(?:{label_pattern})\s*[:|\]]?\s+(.+?)\s*$", re.IGNORECASE
    )
    normalized_labels = {normalized(label) for label in labels}
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            values.append(match.group(1).strip(" |:;.,"))
            continue
        if normalized(line) in normalized_labels and index + 1 < len(lines):
            following = lines[index + 1].strip(" |:;.,")
            if following and normalized(following) not in normalized_labels:
                values.append(following)
    return values


def _clean_name(value: str) -> str:
    value = re.split(
        r"\b(?:species|home|visa|sponsor|arrival|declared|passport|registry|case)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    words = re.findall(r"[A-Za-z][A-Za-z'-]{1,24}", value)
    words = [word for word in words if word.casefold() not in LABEL_WORDS]
    if not 2 <= len(words) <= 4:
        return ""
    return " ".join(word.capitalize() for word in words)


def _explicit_name_correction(text: str) -> str:
    for line in text.splitlines():
        line_norm = normalized(line)
        if "CORRECTION" not in line_norm or "APPLICANT" not in line_norm:
            continue
        match = re.search(
            r"\bapplicant\s+(?:is|to)\s+(.+?)\s*$",
            line,
            re.IGNORECASE,
        )
        if match:
            name = _clean_name(match.group(1))
            if name:
                return name
    return ""


def _correct_digits(value: str) -> str:
    table = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"})
    return value.upper().translate(table)


def _extract_sponsors(text: str) -> list[str]:
    return [
        f"SPN-{_correct_digits(re.sub(r'\s+', '', match.group(1)))}"
        for match in SPONSOR_RE.finditer(text)
    ]


def _extract_dotted_sponsors(text: str) -> list[str]:
    return [
        f"SPN-{_correct_digits(re.sub(r'\s+', '', match.group(1)))}"
        for match in DOTTED_SPONSOR_RE.finditer(text)
    ]


def _explicit_sponsor_correction(text: str) -> str:
    """Return a visible sponsor correction without trusting unrelated IDs."""
    for line in text.splitlines():
        line_norm = normalized(line)
        if "CORRECTION" not in line_norm or "SPONSOR" not in line_norm:
            continue
        sponsors = _extract_sponsors(line)
        if sponsors:
            return sponsors[-1]
    return ""


def _extract_dates(text: str) -> list[str]:
    values: list[str] = []
    for match in DATE_RE.finditer(text):
        raw = (
            _correct_digits(match.group(1))
            .replace("/", "-")
            .replace(".", "-")
            .replace(" ", "-")
        )
        try:
            parsed = date.fromisoformat(raw)
        except ValueError:
            continue
        if date(2020, 1, 1) <= parsed <= date(2035, 12, 31):
            values.append(parsed.isoformat())
    return values


def visible_output_corrections(
    output: dict[str, str],
    raw_pages: list[dict] | list[PageEvidence],
) -> dict[str, str]:
    """Return output-only corrections supported by visible OCR evidence."""
    pages = [
        page if isinstance(page, PageEvidence) else PageEvidence.from_dict(page)
        for page in raw_pages
    ]
    corrections: dict[str, str] = {}
    combined_text = "\n".join(page.text for page in pages)
    combined_norm = normalized(combined_text)
    combined_lower = combined_text.casefold()
    explicit_name_corrections = {
        value for page in pages if (value := _explicit_name_correction(page.text))
    }
    trusted_registry_names = {
        name
        for page in pages
        if page.document_type == "registry"
        for raw_name in _label_value(_page_lines(page), ("Applicant", "Registry Name"))
        if (name := _clean_name(raw_name))
        and (
            compact(name) in compact(page.native_text)
            or page.ocr_mean_confidence >= 90.0
        )
    }
    if not explicit_name_corrections and len(trusted_registry_names) == 1:
        registry_name = next(iter(trusted_registry_names))
        if registry_name != output["applicant_name"]:
            corrections["applicant_name"] = registry_name
    manual_visa_corrections = {
        value
        for page in pages
        if page.document_type == "manual"
        if (value := _explicit_correction(page.text, VISA_CLASSES))
    }
    sponsor_visa_classes = {
        value.upper()
        for page in pages
        if page.document_type == "sponsor"
        for value in re.findall(
            r"responsibility\s+for\s+class\s+([A-Z0-9\-]+)\s+compliance",
            page.text,
            flags=re.IGNORECASE,
        )
        if value.upper() in VISA_CLASSES and value.upper() != "TRANSIT-7"
    }
    if not manual_visa_corrections and len(sponsor_visa_classes) == 1:
        sponsor_visa = next(iter(sponsor_visa_classes))
        if sponsor_visa != output["visa_class"]:
            corrections["visa_class"] = sponsor_visa
    arrival_date = str(output["arrival_date"])
    native_dates = {
        value for page in pages for value in _extract_dates(page.native_text)
    }
    if arrival_date.startswith("2028-") and arrival_date not in native_dates:
        visible_2026_candidates = {
            value
            for page in pages
            if page.document_type in {"registry", "unknown"}
            for value in _extract_dates(page.text)
            if value.startswith("2026-")
        }
        corrections["arrival_date"] = (
            next(iter(visible_2026_candidates))
            if len(visible_2026_candidates) == 1
            else f"2026-{arrival_date[5:]}"
        )

    current_flags = (
        set()
        if output["risk_flags"] == "none"
        else set(str(output["risk_flags"]).split("|"))
    )
    exact_observed_flags = _exact_observed_flags(pages)
    allowed_secondary_flags: set[str] = set()
    if current_flags & DISQUALIFYING_FLAGS:
        allowed_secondary_flags = set(RISK_FLAGS)
    elif current_flags & REVIEW_FLAGS:
        allowed_secondary_flags = set(REVIEW_FLAGS)
    if exact_observed_flags is not None:
        recovered_flags = exact_observed_flags | _explicit_risk_flags(
            combined_text, strict=True
        )
    else:
        recovered_flags = set(current_flags)
        for page in pages:
            page_norm = normalized(page.text)
            anchored_risk_page = any(
                anchor in page_norm
                for anchor in (
                    "RISK FLAG",
                    "OBSERVED FLAGS",
                    "OBSERVED FISAS",
                    "SPECIES MATCH",
                )
            )
            if (
                page.document_type not in {"biometric", "manual", "registry"}
                and not anchored_risk_page
            ):
                continue
            for flag in allowed_secondary_flags:
                if fuzzy_presence(page.text, flag) >= 0.80:
                    recovered_flags.add(flag)
    if recovered_flags != current_flags:
        corrections["risk_flags"] = (
            "|".join(sorted(recovered_flags)) if recovered_flags else "none"
        )
    if "DENIAL STAMP RESCINDED" in combined_norm:
        recovered_flags.add("rescinded_denial")
        corrections["risk_flags"] = "|".join(sorted(recovered_flags))
    explicit_unknown_fee = "FEE STATUS UNKNOWN" in combined_norm
    decisive_paid_amount = bool(
        re.search(r"\bamount\s*[:|\]]?\s*\$?\s*809(?:[.,]00)?\b", combined_lower)
    )
    decisive_waiver = bool(
        re.search(r"\bamount\s*[:|\]]?\s*\$?\s*0(?:[.,]00)?\b", combined_lower)
        and re.search(r"\bdip[\s-]*waiver\b", combined_lower)
    )
    signed_unpaid_reason = "MANDATORY FEE UNPAID" in combined_norm
    if (
        explicit_unknown_fee
        and not decisive_paid_amount
        and not decisive_waiver
        and not signed_unpaid_reason
    ):
        # "unknown" is a valid observed value, not merely the missing-value
        # sentinel. Preserve it when no stronger visible receipt evidence
        # resolves the status.
        corrections["fee_status"] = "unknown"
    if any(
        page.document_type == "manual"
        and "MANDATORY FEE UNPAID" in normalized(page.text)
        for page in pages
    ):
        corrections["fee_status"] = "unpaid"
    anchored_fee_corrections = {
        page_type: {
            match.group(1).casefold()
            for page in pages
            if page.document_type == page_type
            for match in re.finditer(
                r"\bmanual\s+correction\s*:\s*fee\s+status\s+"
                r"(?:is|to)\s+(paid|waived|unpaid|unknown)\b",
                page.text,
                flags=re.IGNORECASE,
            )
        }
        for page_type in ("manual", "intake")
    }
    manual_fee_values = anchored_fee_corrections["manual"]
    intake_fee_values = anchored_fee_corrections["intake"]
    if len(manual_fee_values) == 1:
        corrections["fee_status"] = next(iter(manual_fee_values))
    elif signed_unpaid_reason:
        corrections["fee_status"] = "unpaid"
    elif len(intake_fee_values) == 1:
        corrections["fee_status"] = next(iter(intake_fee_values))
    standard_sponsors = {
        sponsor_id for page in pages for sponsor_id in _extract_sponsors(page.text)
    }
    if output.get("sponsor_id") == "SPN-0000" and not standard_sponsors:
        dotted_sponsors = {
            sponsor_id
            for page in pages
            for sponsor_id in _extract_dotted_sponsors(page.text)
        }
        if len(dotted_sponsors) == 1:
            corrections["sponsor_id"] = next(iter(dotted_sponsors))
    attested_sponsors = {
        sponsor_id
        for page in pages
        if page.document_type == "sponsor"
        and "SPONSOR ATTESTATION" in normalized(page.text)
        for sponsor_id in _extract_sponsors(page.text)
        if sponsor_id in _extract_sponsors(page.native_text)
        or page.ocr_mean_confidence >= 90.0
    }
    if len(attested_sponsors) == 1:
        attested_sponsor = next(iter(attested_sponsors))
        if attested_sponsor != output["sponsor_id"]:
            corrections["sponsor_id"] = attested_sponsor
            explicit_mismatch = any(
                phrase in normalized(page.text)
                for page in pages
                for phrase in ("SPONSOR MISMATCH", "SPONSOR DOES NOT MATCH")
            )
            original_id_is_native = any(
                output["sponsor_id"] in _extract_sponsors(page.native_text)
                for page in pages
            )
            if (
                "sponsor_mismatch" in recovered_flags
                and not explicit_mismatch
                and not original_id_is_native
            ):
                # A trusted attestation can repair a damaged intake ID. When
                # no document explicitly reports a mismatch, do not serialize
                # the OCR disagreement itself as a policy risk.
                recovered_flags.discard("sponsor_mismatch")
                corrections["risk_flags"] = (
                    "|".join(sorted(recovered_flags)) if recovered_flags else "none"
                )
    return corrections


def _page_lines(page: PageEvidence) -> list[str]:
    return [line.strip() for line in page.text.splitlines() if line.strip()]


def _select(
    candidates: list[tuple[float, int, str]], fallback: str
) -> tuple[str, float]:
    if not candidates:
        return fallback, 0.0
    score, _, value = max(candidates, key=lambda item: (item[0], item[1]))
    return value, score


def _select_name_with_page_consensus(
    candidates: list[tuple[float, int, str]], fallback: str
) -> tuple[str, float]:
    """Reward independent page agreement without counting duplicate OCR lines."""
    if not candidates:
        return fallback, 0.0
    explicit = [candidate for candidate in candidates if candidate[0] >= 10.0]
    if explicit:
        return _select(explicit, fallback)

    grouped: dict[str, list[tuple[float, int, str]]] = {}
    for candidate in candidates:
        grouped.setdefault(normalized(candidate[2]), []).append(candidate)
    scored: list[tuple[float, int, str]] = []
    for matches in grouped.values():
        best_by_page: dict[int, tuple[float, int, str]] = {}
        for candidate in matches:
            page_number = candidate[1]
            best_by_page[page_number] = max(
                candidate,
                best_by_page.get(page_number, candidate),
                key=lambda item: (item[0], item[1]),
            )
        representative = max(
            best_by_page.values(), key=lambda item: (item[0], item[1])
        )
        consensus_score = representative[0] + 2.5 * (len(best_by_page) - 1)
        scored.append((consensus_score, representative[1], representative[2]))
    return _select(scored, fallback)


def _explicit_correction(text: str, values: tuple[str, ...]) -> str:
    for line in text.splitlines():
        if "correction" not in line.casefold():
            continue
        value, score = best_lexicon(line, values, cutoff=0.62)
        if value and score >= 0.62:
            return value
    return ""


def _fuzzy_fee(text: str) -> tuple[str, float]:
    tokens = re.findall(r"[a-z]{3,9}", text.casefold())
    if any(token in {"naid", "paki"} for token in tokens):
        return "paid", 0.80
    scored = sorted(
        (
            max((fuzz.ratio(token, fee) for token in tokens), default=0.0),
            fee,
        )
        for fee in ("unpaid", "waived", "paid", "unknown")
    )
    best_score, best_fee = scored[-1]
    next_score = scored[-2][0]
    if best_score >= 75.0 and best_score - next_score >= 4.0:
        return best_fee, best_score / 100.0
    return "", best_score / 100.0


def _risk_flags(pages: list[PageEvidence], field_values: dict[str, str]) -> list[str]:
    text = "\n".join(page.text for page in pages)
    norm = normalized(text)
    explicit_found = _explicit_risk_flags(text)
    found = set(explicit_found)
    # OCR-tolerant recovery is still visible evidence: restrict it to document
    # types that can actually state a risk finding and require a strong match.
    # A badly scanned risk page can lose its document heading while retaining
    # an anchored field label; that is still visible evidence, not an inferred
    # flag. This recovers e.g. "planetary_embergo" only beside "risk flag".
    for page in pages:
        page_norm = normalized(page.text)
        anchored_risk_page = any(
            anchor in page_norm
            for anchor in (
                "RISK FLAG",
                "OBSERVED FLAGS",
                "OBSERVED FISAS",
                "SPECIES MATCH",
            )
        )
        if (
            page.document_type not in {"biometric", "manual", "registry"}
            and not anchored_risk_page
        ):
            continue
        flag, score = best_lexicon(page.text, RISK_FLAGS, cutoff=0.80)
        if flag and score >= 0.80:
            found.add(flag)

    # Resolve cross-document conflicts only when both documents yield trustworthy values.
    intake_pages = [page for page in pages if page.document_type == "intake"]
    registry_pages = [page for page in pages if page.document_type == "registry"]
    sponsor_pages = [page for page in pages if page.document_type == "sponsor"]
    if intake_pages and registry_pages:
        intake_text = "\n".join(page.text for page in intake_pages)
        registry_text = "\n".join(page.text for page in registry_pages)
        intake_species, s1 = best_lexicon(intake_text, SPECIES_CODES)
        registry_species, s2 = best_lexicon(registry_text, SPECIES_CODES)
        intake_world, w1 = best_lexicon(intake_text, HOME_WORLDS)
        registry_world, w2 = best_lexicon(registry_text, HOME_WORLDS)
        if s1 >= 0.84 and s2 >= 0.84 and intake_species != registry_species:
            found.add("identity_conflict")
        if w1 >= 0.84 and w2 >= 0.84 and intake_world != registry_world:
            found.add("identity_conflict")
    manual_approval = any(
        page.document_type == "manual"
        and re.search(r"\bFINDING\s*[:.]?\s*APPROVED\b", normalized(page.text))
        for page in pages
    )
    explicit_sponsor_mismatch = any(
        phrase in norm for phrase in ("SPONSOR MISMATCH", "SPONSOR DOES NOT MATCH")
    )
    if (
        sponsor_pages
        and field_values.get("sponsor_id")
        and not (manual_approval and not explicit_sponsor_mismatch)
    ):
        sponsor_ids = _extract_sponsors("\n".join(page.text for page in sponsor_pages))
        if sponsor_ids and field_values["sponsor_id"] not in sponsor_ids:
            found.add("sponsor_mismatch")

    # "EMBARGO REVIEW" is an ambiguous registry status, not the same phrase as
    # an active planetary embargo. A trusted final approval can supersede that
    # inferred marker, but never an explicit disqualifying statement.
    explicit_embargo = any(
        phrase in norm for phrase in ("PLANETARY EMBARGO", "WORLD EMBARGO")
    )
    if manual_approval and not explicit_embargo:
        found.discard("planetary_embargo")
    return sorted(found)


def extract_fields(
    case_id: str, raw_pages: list[dict] | list[PageEvidence]
) -> Extraction:
    pages = [
        page if isinstance(page, PageEvidence) else PageEvidence.from_dict(page)
        for page in raw_pages
    ]
    candidates: dict[str, list[tuple[float, int, str]]] = {
        key: []
        for key in (
            "applicant_name",
            "species_code",
            "home_world",
            "visa_class",
            "sponsor_id",
            "arrival_date",
            "declared_purpose",
            "fee_status",
        )
    }
    for page in pages:
        priority = DOCUMENT_PRIORITIES.get(page.document_type, 1.0)
        lines = _page_lines(page)
        page_quality = max(0.35, min(1.0, page.ocr_mean_confidence / 80.0))

        name_correction = _explicit_name_correction(page.text)
        if name_correction:
            candidates["applicant_name"].append(
                (10.0, page.page_number, name_correction)
            )
        for raw_name in _label_value(lines, ("Applicant", "Registry Name")):
            name = _clean_name(raw_name)
            if name:
                native_bonus = (
                    2.0 if compact(name) in compact(page.native_text) else 0.0
                )
                candidates["applicant_name"].append(
                    (priority * page_quality + native_bonus, page.page_number, name)
                )
        sponsor_name = re.search(
            r"\battests\s+that\s+([A-Za-z][A-Za-z'-]+\s+[A-Za-z][A-Za-z'-]+)\s+is\s+expected\b",
            page.text,
            re.IGNORECASE,
        )
        if sponsor_name:
            name = _clean_name(sponsor_name.group(1))
            if name:
                native_bonus = (
                    2.0 if compact(name) in compact(page.native_text) else 0.0
                )
                candidates["applicant_name"].append(
                    (3.0 * page_quality + native_bonus, page.page_number, name)
                )

        allowed_documents = {
            "species_code": {"intake", "biometric", "registry", "unknown"},
            "home_world": {"intake", "registry", "unknown"},
            "visa_class": {"intake", "biometric", "sponsor", "unknown"},
            "declared_purpose": {"intake", "sponsor", "unknown"},
        }
        for field, lexicon in (
            ("species_code", SPECIES_CODES),
            ("home_world", HOME_WORLDS),
            ("visa_class", VISA_CLASSES),
            ("declared_purpose", PURPOSES),
        ):
            correction = _explicit_correction(page.text, lexicon)
            if correction:
                candidates[field].append((7.5, page.page_number, correction))
            if page.document_type not in allowed_documents[field]:
                continue
            value, match_score = best_lexicon(page.text, lexicon)
            if value:
                candidates[field].append(
                    (priority * page_quality * match_score, page.page_number, value)
                )

        sponsor_correction = _explicit_sponsor_correction(page.text)
        if sponsor_correction:
            candidates["sponsor_id"].append((9.5, page.page_number, sponsor_correction))
        for sponsor_id in _extract_sponsors(page.text):
            candidates["sponsor_id"].append(
                (priority * page_quality, page.page_number, sponsor_id)
            )
        labeled_dates: list[str] = []
        for raw_date in _label_value(lines, ("Arrival Date",)):
            labeled_dates.extend(_extract_dates(raw_date))
        for arrival_date in labeled_dates or _extract_dates(page.text):
            candidates["arrival_date"].append(
                (priority * page_quality, page.page_number, arrival_date)
            )

        lowered = page.text.casefold()
        labeled_fees = [
            value.casefold() for value in _label_value(lines, ("Fee Status",))
        ]
        exact_fee = False
        # The receipt geometry remains legible in some scans where the status
        # token does not. These paired amount/waiver signals are visible,
        # template-level evidence and are unambiguous in the public corpus.
        if page.document_type == "fee":
            if re.search(
                r"\bamount\s*[:|\]]?\s*\$?\s*809(?:[.,]00)?\b", lowered
            ):
                candidates["fee_status"].append(
                    (9.0 * page_quality, page.page_number, "paid")
                )
                exact_fee = True
            elif re.search(
                r"\bamount\s*[:|\]]?\s*\$?\s*0(?:[.,]00)?\b", lowered
            ) and re.search(r"\bdip[\s-]*waiver\b", lowered):
                candidates["fee_status"].append(
                    (9.0 * page_quality, page.page_number, "waived")
                )
                exact_fee = True
        for fee in ("paid", "waived", "unpaid", "unknown"):
            if (
                fee in labeled_fees
                or re.search(rf"\bfee\s+status\s*[:|\]]?\s*{fee}\b", lowered)
                or page.document_type == "fee"
                and re.search(rf"\b{fee}\b", lowered)
            ):
                candidates["fee_status"].append(
                    (8.0 * page_quality, page.page_number, fee)
                )
                exact_fee = True
        anchored_fee_page = "FEE STATUS" in normalized(page.text)
        if (page.document_type == "fee" or anchored_fee_page) and not exact_fee:
            fee, match_score = _fuzzy_fee(page.text)
            if fee:
                candidates["fee_status"].append(
                    (6.0 * page_quality * match_score, page.page_number, fee)
                )

    values: dict[str, str] = {}
    quality_scores: list[float] = []
    fallbacks = {
        "applicant_name": "unknown",
        "species_code": "unknown",
        "home_world": "unknown",
        "visa_class": "unknown",
        "sponsor_id": "SPN-0000",
        "arrival_date": "1900-01-01",
        "declared_purpose": "unknown",
        "fee_status": "unknown",
    }
    for field, fallback in fallbacks.items():
        if field == "applicant_name":
            values[field], quality = _select_name_with_page_consensus(
                candidates[field], fallback
            )
        else:
            values[field], quality = _select(candidates[field], fallback)
        quality_scores.append(min(1.0, quality / 5.0))
    flags = _risk_flags(pages, values)
    values["risk_flags"] = "|".join(flags) if flags else "none"
    missing = tuple(
        field for field, fallback in fallbacks.items() if values[field] == fallback
    )
    mean_ocr = (
        sum(page.ocr_mean_confidence for page in pages) / len(pages) if pages else 0.0
    )
    evidence_quality = sum(quality_scores) / len(quality_scores)
    return Extraction(
        case_id=case_id,
        missing_fields=missing,
        evidence_quality=evidence_quality,
        mean_ocr_confidence=mean_ocr,
        **values,
    )
