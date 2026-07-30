"""Field extraction from trusted (visible) evidence.

v0 sources, in field-manual precedence order:
  - visible text-layer labels on scan pages (biometric slip: Applicant/Species/flags)
  - native-text pages (sponsor attestation letters)
Pages naming a different case id than the packet's are decoy pages for another
applicant and are ignored entirely (multi-applicant trap).

OCR of the page rasters is layered on top for image-only fields.
"""
import re

from .vocab import (CASE_RE, DATE_RE, FEES, FLAGS, PURPOSES, SPECIES, SPONSOR_RE,
                    VISAS, WORLDS, snap)

LABEL_PATTERNS = {
    "applicant_name": re.compile(r"Applicant(?: name)?:\s*([A-Z][a-z]+ [A-Z][a-z]+)"),
    "species_code": re.compile(r"Species(?: Match| code)?:\s*([A-Z_ ]{4,})"),
    "risk_flags": re.compile(r"(?:Observed|Risk) flags?:\s*([a-z_|]+|none)"),
    "home_world": re.compile(r"Home ?world:\s*(\S[^\n]*)"),
    "visa_class": re.compile(r"Visa(?: class)?:\s*([A-Z]{2,8}-?\d?)"),
    "arrival_date": re.compile(r"Arrival(?: date)?:\s*(\d{4}-\d{2}-\d{2})"),
    "fee_status": re.compile(r"Fee(?: status)?:\s*([a-z]+)"),
    "declared_purpose": re.compile(r"Purpose:\s*([a-z ]+)"),
}
# Sponsor-letter sentence parsed as INDEPENDENT per-field patterns: the
# original all-in-one regex silently dropped sponsor+purpose+visa whenever one
# group failed (e.g. the applicant printed as "[NAME CUT OUT]").
# \s+ (not literal spaces) throughout: the letter sentence wraps across lines
# mid-name and mid-purpose ("…for medical\nconsult.").
LETTER_SPONSOR_RE = re.compile(r"Sponsor\s+(SPN-\d{4})\s+attests")
LETTER_NAME_RE = re.compile(r"attests\s+that\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+is\s+expected")
LETTER_PURPOSE_RE = re.compile(r"is\s+expected\s+on\s+Earth\s+for\s+([a-z\s]+?)\.")
LETTER_VISA_RE = re.compile(r"class\s+([A-Z]{2,8}-\d)\s+compliance")


_FOOTER_RE = re.compile(r"(?i)packet\s*MIB[\s-]*\d{6}\s*/?\s*page")


def page_case_ids(text):
    """Case ids named in body text, excluding packet footer lines
    (space-tolerant: OCR emits "PacketMIB-000320/page2")."""
    body = "\n".join(l for l in text.splitlines() if not _FOOTER_RE.search(l))
    return set(CASE_RE.findall(body))


def extract_from_visible_text(case_id, page_texts):
    """page_texts: list of visible-span text per page. Returns {field: (value, source)}."""
    out = {}

    def put(field, value, source):
        if field not in out and value:
            out[field] = (value.strip(), source)

    for text in page_texts:
        ids = page_case_ids(text)
        if ids and case_id not in ids:
            continue  # decoy page for a different applicant
        is_letter = "attests" in text

        if is_letter:
            m = LETTER_SPONSOR_RE.search(text)
            if m:
                put("sponsor_id", m.group(1), "sponsor_letter")
            m = LETTER_NAME_RE.search(text)
            if m:
                put("applicant_name", re.sub(r"\s+", " ", m.group(1)), "sponsor_letter")
            m = LETTER_PURPOSE_RE.search(text)
            if m:
                purpose, _, _ = snap(re.sub(r"\s+", " ", m.group(1)).strip(), PURPOSES)
                put("declared_purpose", purpose, "sponsor_letter")
            m = LETTER_VISA_RE.search(text)
            if m:
                visa, _, _ = snap(m.group(1), VISAS)
                put("visa_class", visa, "sponsor_letter")

        for field, pattern in LABEL_PATTERNS.items():
            m = pattern.search(text)
            if not m:
                continue
            value = m.group(1).strip()
            source = "letter_label" if is_letter else "slip_label"
            if field == "species_code":
                value, _, _ = snap(value, SPECIES)
            elif field == "home_world":
                value, _, _ = snap(value, WORLDS)
            elif field == "visa_class":
                value, _, _ = snap(value, VISAS)
            elif field == "fee_status":
                value, _, _ = snap(value, FEES)
            elif field == "declared_purpose":
                value, _, _ = snap(value, PURPOSES)
            elif field == "risk_flags":
                parts = [snap(p, FLAGS)[0] for p in value.split("|") if p and p != "none"]
                value = "|".join(sorted(p for p in parts if p)) or "none"
            put(field, value, source)
    return out
