from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date

from rapidfuzz import fuzz

from .constants import (
    DOCUMENT_PRIORITIES,
    HOME_WORLDS,
    PURPOSES,
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

    def output_fields(self) -> dict:
        value = asdict(self)
        value.pop("missing_fields")
        value.pop("evidence_quality")
        value.pop("mean_ocr_confidence")
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


def best_lexicon(text: str, values: tuple[str, ...], cutoff: float = 0.72) -> tuple[str, float]:
    scored = [(value, fuzzy_presence(text, value)) for value in values]
    value, score = max(scored, key=lambda item: item[1])
    return (value, score) if score >= cutoff else ("", score)


def _label_value(lines: list[str], labels: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(rf"^\s*(?:{label_pattern})\s*[:|\]]?\s+(.+?)\s*$", re.IGNORECASE)
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


def _correct_digits(value: str) -> str:
    table = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"})
    return value.upper().translate(table)


def _extract_sponsors(text: str) -> list[str]:
    return [
        f"SPN-{_correct_digits(re.sub(r'\s+', '', match.group(1)))}"
        for match in SPONSOR_RE.finditer(text)
    ]


def _extract_dates(text: str) -> list[str]:
    values: list[str] = []
    for match in DATE_RE.finditer(text):
        raw = _correct_digits(match.group(1)).replace("/", "-").replace(".", "-").replace(" ", "-")
        try:
            parsed = date.fromisoformat(raw)
        except ValueError:
            continue
        if date(2020, 1, 1) <= parsed <= date(2035, 12, 31):
            values.append(parsed.isoformat())
    return values


def _page_lines(page: PageEvidence) -> list[str]:
    return [line.strip() for line in page.text.splitlines() if line.strip()]


def _select(candidates: list[tuple[float, int, str]], fallback: str) -> tuple[str, float]:
    if not candidates:
        return fallback, 0.0
    score, _, value = max(candidates, key=lambda item: (item[0], item[1]))
    return value, score


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
    scored = sorted(
        (
            max((fuzz.ratio(token, fee) for token in tokens), default=0.0),
            fee,
        )
        for fee in ("unpaid", "waived", "paid", "unknown")
    )
    best_score, best_fee = scored[-1]
    next_score = scored[-2][0]
    if best_score >= 76.0 and best_score - next_score >= 4.0:
        return best_fee, best_score / 100.0
    return "", best_score / 100.0


def _risk_flags(pages: list[PageEvidence], field_values: dict[str, str]) -> list[str]:
    text = "\n".join(page.text for page in pages)
    norm = normalized(text)
    found: set[str] = set()
    phrases = {
        "active_warrant": ("ACTIVE WARRANT", "WARRANT ACTIVE"),
        "biohazard_red": ("BIOHAZARD RED", "BIO HAZARD RED", "HAZARD LEVEL RED"),
        "identity_conflict": ("IDENTITY CONFLICT", "IDENTITY MISMATCH"),
        "illegible_biometrics": (
            "ILLEGIBLE BIOMETRICS",
            "BIOMETRICS ILLEGIBLE",
            "BIOMETRIC UNREADABLE",
            "BIOMETRICS UNREADABLE",
        ),
        "memory_tampering": ("MEMORY TAMPERING", "MEMORY ALTERATION"),
        "planetary_embargo": ("PLANETARY EMBARGO", "REGISTRY STATUS EMBARGO", "WORLD EMBARGO"),
        "rescinded_denial": ("RESCINDED DENIAL", "DENIAL RESCINDED", "DENIAL CROSSED OUT"),
        "sponsor_mismatch": ("SPONSOR MISMATCH", "SPONSOR DOES NOT MATCH"),
    }
    for flag in RISK_FLAGS:
        if normalized(flag) in norm:
            found.add(flag)
            continue
        if any(phrase in norm for phrase in phrases[flag]):
            found.add(flag)
    # OCR-tolerant recovery is still visible evidence: restrict it to document
    # types that can actually state a risk finding and require a strong match.
    for page in pages:
        if page.document_type not in {"biometric", "manual", "registry"}:
            continue
        flag, score = best_lexicon(page.text, RISK_FLAGS, cutoff=0.84)
        if flag and score >= 0.84:
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
    if sponsor_pages and field_values.get("sponsor_id"):
        sponsor_ids = _extract_sponsors("\n".join(page.text for page in sponsor_pages))
        if sponsor_ids and field_values["sponsor_id"] not in sponsor_ids:
            found.add("sponsor_mismatch")
    return sorted(found)


def extract_fields(case_id: str, raw_pages: list[dict] | list[PageEvidence]) -> Extraction:
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

        for raw_name in _label_value(lines, ("Applicant", "Registry Name")):
            name = _clean_name(raw_name)
            if name:
                native_bonus = 2.0 if compact(name) in compact(page.native_text) else 0.0
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
                native_bonus = 2.0 if compact(name) in compact(page.native_text) else 0.0
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

        for sponsor_id in _extract_sponsors(page.text):
            candidates["sponsor_id"].append((priority * page_quality, page.page_number, sponsor_id))
        labeled_dates: list[str] = []
        for raw_date in _label_value(lines, ("Arrival Date",)):
            labeled_dates.extend(_extract_dates(raw_date))
        for arrival_date in labeled_dates or _extract_dates(page.text):
            candidates["arrival_date"].append((priority * page_quality, page.page_number, arrival_date))

        lowered = page.text.casefold()
        labeled_fees = [value.casefold() for value in _label_value(lines, ("Fee Status",))]
        exact_fee = False
        for fee in ("paid", "waived", "unpaid", "unknown"):
            if (
                fee in labeled_fees
                or re.search(rf"\bfee\s+status\s*[:|\]]?\s*{fee}\b", lowered)
                or page.document_type == "fee" and re.search(rf"\b{fee}\b", lowered)
            ):
                candidates["fee_status"].append((8.0 * page_quality, page.page_number, fee))
                exact_fee = True
        if page.document_type == "fee" and not exact_fee:
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
        values[field], quality = _select(candidates[field], fallback)
        quality_scores.append(min(1.0, quality / 5.0))
    flags = _risk_flags(pages, values)
    values["risk_flags"] = "|".join(flags) if flags else "none"
    missing = tuple(field for field, fallback in fallbacks.items() if values[field] == fallback)
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
