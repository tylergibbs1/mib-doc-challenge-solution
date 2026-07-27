from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
from rapidfuzz import fuzz

from .constants import (
    ADJUDICATIONS,
    DISQUALIFYING_FLAGS,
    EMBARGOED_HOME_WORLDS,
    REVIEW_FLAGS,
    REVOKED_SPONSORS,
)
from .fields import Extraction, normalized, visible_output_corrections
from .pdf import PageEvidence

CONFIDENCE_GUARDRAIL_SOURCES = (
    "none",
    "visible_manual_finding",
    "disqualifying_risk",
    "embargoed_home_world",
    "revoked_sponsor",
    "transit_class",
    "stale_non_diplomatic_packet",
    "unpaid_fee",
    "combined_review_flags",
    "emitted_disqualifying_risk",
    "emitted_embargoed_home_world",
    "emitted_revoked_sponsor",
    "emitted_transit_class",
    "emitted_stale_non_diplomatic_packet",
    "emitted_unpaid_fee",
    "emitted_unknown_fee",
    "emitted_combined_review_flags",
    "emitted_review_risk",
    "emitted_missing_policy_evidence",
    "visible_dip_layout_consensus",
)
CONFIDENCE_DOCUMENT_TYPES = (
    "manual",
    "intake",
    "biometric",
    "fee",
    "sponsor",
    "registry",
    "unknown",
)


@dataclass
class Decision:
    adjudication: str
    confidence: float
    source: str
    probabilities: dict[str, float]


def _visible_text_from(pages: list[PageEvidence], *, model_view: bool) -> str:
    # Case identifiers are output keys, not evidence. Removing them prevents
    # character n-grams from learning accidental train-only identity features.
    return "\n\n".join(
        f"[PAGE {page.page_number} {page.document_type.upper()}]\n"
        + re.sub(
            r"\bMIB[-\s]\d{6}\b",
            "MIB-CASE",
            page.model_text if model_view and page.model_text else page.text,
            flags=re.IGNORECASE,
        )
        for page in pages
    )


def visible_text(pages: list[PageEvidence]) -> str:
    return _visible_text_from(pages, model_view=False)


def visible_model_text(pages: list[PageEvidence]) -> str:
    """Return stable baseline OCR for learned models, preserving best OCR for rules."""
    return _visible_text_from(pages, model_view=True)


def _manual_evidence_pages(pages: list[PageEvidence]) -> list[PageEvidence]:
    """Include notes whose damaged heading defeated the page-type classifier."""
    manual_pages: list[PageEvidence] = []
    expected_heading = "MANUAL ADJUDICATOR NOTE"
    for page in pages:
        if page.document_type == "manual":
            manual_pages.append(page)
            continue
        heading = " ".join(normalized(page.text).split()[:20])
        if fuzz.partial_ratio(heading, expected_heading) >= 60.0:
            manual_pages.append(page)
    return manual_pages


def explicit_manual_finding(pages: list[PageEvidence]) -> tuple[str, float]:
    candidates: list[tuple[int, float, str]] = []
    manual_pages = _manual_evidence_pages(pages)
    for page in manual_pages:
        exact_on_finding_line = False
        for line in page.text.splitlines():
            line_norm = normalized(line)
            if "FINDING" not in line_norm:
                continue
            for label in ADJUDICATIONS:
                if label.replace("_", " ") in line_norm:
                    candidates.append((page.page_number, 1.0, label))
                    exact_on_finding_line = True
        if exact_on_finding_line:
            continue
        norm = normalized(page.text)
        exact_labels = [
            label for label in ADJUDICATIONS if label.replace("_", " ") in norm
        ]
        if len(exact_labels) == 1:
            candidates.append((page.page_number, 0.9, exact_labels[0]))
    if not candidates:
        # On damaged notes the finding line can be destroyed while the signed
        # reason remains readable. Use only policy-specific reason phrases,
        # never generic words such as "denial" that may occur in a watermark.
        reason_markers = {
            "APPROVED": (
                "APPROVAL SUPPORTED BY SURVIVING VISIBLE EVIDENCE",
                "CLEAN OR EXCEPTION QUALIFIED PACKET",
            ),
            "DENIED": (
                "DENIAL SUPPORTED BY DAMAGED REGISTRY EVIDENCE",
                "MANDATORY FEE UNPAID",
                "REVOKED SPONSOR",
                "DISQUALIFYING RISK FLAG",
            ),
            "NEEDS_REVIEW": (
                "ARRIVAL DATE MISSING FROM TRUSTED VISIBLE EVIDENCE",
                "DAMAGED OR CONTRADICTORY VISIBLE EVIDENCE",
                "FEE STATUS UNKNOWN",
                "RESCINDED DENIAL",
                "REVIEW ONLY RISK FLAG",
            ),
        }
        for page in manual_pages:
            norm = normalized(page.text)
            matched = [
                label
                for label, markers in reason_markers.items()
                if any(marker in norm for marker in markers)
            ]
            if len(matched) == 1:
                candidates.append((page.page_number, 0.88, matched[0]))
    if not candidates:
        # Reason phrases can remain nearly complete even when OCR damages a
        # few glyphs and destroys the finding line. Match the full
        # policy-specific phrase and require a wide runner-up margin.
        for page in manual_pages:
            norm = normalized(page.text)
            scores = sorted(
                (
                    max(fuzz.partial_ratio(norm, marker) for marker in markers),
                    label,
                )
                for label, markers in reason_markers.items()
            )
            best_score, best_label = scores[-1]
            next_score = scores[-2][0]
            if best_score >= 85.0 and best_score - next_score >= 10.0:
                candidates.append(
                    (page.page_number, 0.80 + 0.0012 * best_score, best_label)
                )
    if not candidates:
        # A damaged scan often preserves the layout and most glyphs in the
        # signed finding line while corrupting one or two characters. Match the
        # complete "finding + decision" phrase, require a useful line length,
        # and require separation from the runner-up so a bare "Finding:" never
        # becomes an invented decision.
        for page in manual_pages:
            for line in page.text.splitlines():
                line_norm = normalized(line)
                if len(line_norm) < 8:
                    continue
                scores = sorted(
                    (
                        fuzz.partial_ratio(
                            line_norm, f"FINDING {label.replace('_', ' ')}"
                        ),
                        label,
                    )
                    for label in ADJUDICATIONS
                )
                best_score, best_label = scores[-1]
                next_score = scores[-2][0]
                if best_score >= 78.0 and best_score - next_score >= 3.0:
                    candidates.append(
                        (page.page_number, 0.78 + 0.002 * best_score, best_label)
                    )
                    break
    if not candidates:
        return "", 0.0
    _, score, label = max(candidates)
    return label, score


def _flags(record: Extraction) -> set[str]:
    return set() if record.risk_flags == "none" else set(record.risk_flags.split("|"))


def deterministic_guardrail(
    record: Extraction, pages: list[PageEvidence]
) -> tuple[str, float, str] | None:
    manual, manual_score = explicit_manual_finding(pages)
    text = normalized(visible_text(pages))
    if manual:
        return manual, 0.82 + 0.16 * manual_score, "visible_manual_finding"

    flags = _flags(record)
    if flags & DISQUALIFYING_FLAGS:
        return "DENIED", 0.96, "disqualifying_risk"
    if (
        record.home_world in EMBARGOED_HOME_WORLDS
        and record.visa_class != "DIP-1"
    ):
        return "DENIED", 0.96, "embargoed_home_world"
    if record.sponsor_id in REVOKED_SPONSORS and record.visa_class != "DIP-1":
        return "DENIED", 0.96, "revoked_sponsor"
    if record.visa_class == "TRANSIT-7":
        return "DENIED", 0.94, "transit_class"
    try:
        arrival = date.fromisoformat(record.arrival_date)
    except ValueError:
        arrival = None
    if arrival and arrival < date(2026, 1, 1) and record.visa_class != "DIP-1":
        return "DENIED", 0.93, "stale_non_diplomatic_packet"
    waiver_visible = "HARDSHIP WAIVER" in text or "WAIVER APPROVED" in text
    if record.fee_status == "unpaid" and not waiver_visible:
        return "DENIED", 0.95, "unpaid_fee"
    return None


def evidence_policy(
    record: Extraction, pages: list[PageEvidence]
) -> tuple[str, float, str]:
    guardrail = deterministic_guardrail(record, pages)
    if guardrail:
        return guardrail

    flags = _flags(record)
    text = normalized(visible_text(pages))
    if flags & REVIEW_FLAGS:
        return "NEEDS_REVIEW", 0.88, "visible_review_risk"
    if record.fee_status == "unknown":
        return "NEEDS_REVIEW", 0.86, "unknown_fee"
    waiver_visible = "HARDSHIP WAIVER" in text or "WAIVER APPROVED" in text
    if (
        record.fee_status == "waived"
        and record.visa_class != "DIP-1"
        and not waiver_visible
    ):
        return "NEEDS_REVIEW", 0.84, "unsupported_fee_waiver"
    core_missing = {"visa_class", "arrival_date"}
    if record.visa_class != "DIP-1":
        core_missing.add("sponsor_id")
    if core_missing & set(record.missing_fields):
        return "NEEDS_REVIEW", 0.82, "missing_policy_evidence"
    return "APPROVED", 0.88, "complete_policy_evidence"


def emitted_policy_guardrail(
    record: Extraction,
    output: dict,
    pages: list[PageEvidence],
) -> tuple[str, float, str] | None:
    """Prevent serialized fields from contradicting the serialized decision.

    Output-only priors and high-confidence field corrections improve extraction,
    but they may change a policy-bearing value after the original record was
    adjudicated. This second pass is deliberately one-way: it can only make an
    approval more conservative.
    """
    # A signed manual finding is the highest-precedence visible evidence and
    # deterministic_guardrail() has already applied it. Output-only priors and
    # statistical field corrections are lower-precedence estimates; allowing
    # them to overwrite the signed finding reverses the published evidence
    # hierarchy and makes damaged manual packets systematically less reliable.
    manual, _ = explicit_manual_finding(pages)
    if manual:
        return None

    output_flags = (
        set()
        if output["risk_flags"] == "none"
        else set(str(output["risk_flags"]).split("|"))
    )
    visa_class = str(output["visa_class"])
    home_world = str(output["home_world"])
    sponsor_id = str(output["sponsor_id"])
    fee_status = str(output["fee_status"])
    text = normalized(visible_text(pages))

    if output_flags & DISQUALIFYING_FLAGS:
        return "DENIED", 0.96, "emitted_disqualifying_risk"
    if home_world in EMBARGOED_HOME_WORLDS and visa_class != "DIP-1":
        return "DENIED", 0.96, "emitted_embargoed_home_world"
    if sponsor_id in REVOKED_SPONSORS and visa_class != "DIP-1":
        return "DENIED", 0.96, "emitted_revoked_sponsor"
    if visa_class == "TRANSIT-7":
        return "DENIED", 0.94, "emitted_transit_class"

    arrival_text = str(output["arrival_date"])
    try:
        arrival = date.fromisoformat(arrival_text)
    except ValueError:
        arrival = None
    if (
        arrival
        and arrival != date(1900, 1, 1)
        and arrival < date(2026, 1, 1)
        and visa_class != "DIP-1"
    ):
        return "DENIED", 0.93, "emitted_stale_non_diplomatic_packet"

    waiver_visible = "HARDSHIP WAIVER" in text or "WAIVER APPROVED" in text
    if fee_status == "unpaid" and not waiver_visible:
        return "DENIED", 0.95, "emitted_unpaid_fee"
    if fee_status == "unknown":
        return "NEEDS_REVIEW", 0.86, "emitted_unknown_fee"
    if output_flags & REVIEW_FLAGS:
        return "NEEDS_REVIEW", 0.88, "emitted_review_risk"

    missing_fields = set(record.missing_fields)
    if arrival_text == "1900-01-01" or "arrival_date" in missing_fields:
        return "NEEDS_REVIEW", 0.82, "emitted_missing_policy_evidence"
    if "visa_class" in missing_fields:
        return "NEEDS_REVIEW", 0.82, "emitted_missing_policy_evidence"
    if visa_class != "DIP-1" and "sponsor_id" in missing_fields:
        return "NEEDS_REVIEW", 0.82, "emitted_missing_policy_evidence"

    if record.fee_status == "unknown":
        return "NEEDS_REVIEW", 0.82, "emitted_missing_policy_evidence"
    return None


def _clean_visible_person_name(raw: str) -> str:
    """Return a conservative two-token name from a layout-text field."""
    text = " ".join(raw.split())
    text = re.split(
        r"\s{2,}|\s+(?:PASSPORT|CASE|SPN|VISA|SPECIES|HOME|ARRIVAL|FEE)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    parts = text.split()
    if len(parts) < 2:
        return ""
    candidate = " ".join(parts[:2])
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'-]+", part) for part in parts[:2]):
        return ""
    return candidate.casefold()


def visible_approval_guardrail(
    record: Extraction,
    output: dict,
    pages: list[PageEvidence],
) -> tuple[str, float, str] | None:
    """Certify a narrow diplomatic approval from corroborated visible facts.

    This positive rule requires the paid amount itself and the same applicant
    identity on intake and registry evidence. It never treats absent risk text
    as clean and never overrides a signed adjudicator finding.
    """
    manual, _ = explicit_manual_finding(pages)
    if manual:
        return None
    if str(output["visa_class"]) != "DIP-1":
        return None
    if str(output["fee_status"]) != "paid":
        return None
    if str(output["risk_flags"]) != "none":
        return None
    if str(output["arrival_date"]) == "1900-01-01":
        return None
    if str(output["declared_purpose"]) == "medical consult":
        return None

    text = visible_text(pages)
    if not re.search(
        r"(?:Amount\s*[:|]?\s*)?\$\s*809(?:[.,]00)?",
        text,
        flags=re.IGNORECASE,
    ):
        return None
    registries = {
        cleaned
        for raw in re.findall(
            r"Registry\s+Name\s*:?\s*"
            r"([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)+)",
            text,
            flags=re.IGNORECASE,
        )
        if (cleaned := _clean_visible_person_name(raw))
    }
    applicants = {
        cleaned
        for raw in re.findall(
            r"Applicant\s*:?\s*"
            r"([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)+)",
            text,
            flags=re.IGNORECASE,
        )
        if (cleaned := _clean_visible_person_name(raw))
    }
    if len(registries) != 1 or registries != applicants:
        return None
    return "APPROVED", 0.61, "visible_dip_layout_consensus"


def expected_utility_label(probabilities: np.ndarray, classes: list[str]) -> str:
    probability = {
        label: float(probabilities[index]) for index, label in enumerate(classes)
    }
    approved = (
        8.0 * probability["APPROVED"]
        - 4.0 * probability["DENIED"]
        + probability["NEEDS_REVIEW"]
    )
    denied = 8.0 * probability["DENIED"] + probability["NEEDS_REVIEW"]
    review = (
        2.0 * probability["APPROVED"]
        + 2.0 * probability["DENIED"]
        + 8.0 * probability["NEEDS_REVIEW"]
    )
    return max(
        (
            (approved, "APPROVED"),
            (denied, "DENIED"),
            (review, "NEEDS_REVIEW"),
        )
    )[1]


def threshold_policy_label(
    probabilities: np.ndarray,
    classes: list[str],
    policy: dict[str, float],
) -> str:
    probability = {
        label: float(probabilities[index]) for index, label in enumerate(classes)
    }
    if (
        probability["APPROVED"] >= policy["approve_threshold"]
        and probability["APPROVED"] - probability["DENIED"] >= policy["approve_margin"]
    ):
        return "APPROVED"
    if probability["DENIED"] >= policy["deny_threshold"]:
        return "DENIED"
    return "NEEDS_REVIEW"


def confidence_feature_row(
    probabilities: np.ndarray,
    adjudication: str,
    raw_confidence: float,
    disagreement: float,
    record: Extraction,
    pages: list[PageEvidence],
    guardrail: tuple[str, float, str] | None,
    classes: list[str],
) -> np.ndarray:
    """Build leakage-safe confidence features shared by training and inference."""
    ordered = np.sort(probabilities)
    margin = float(ordered[-1] - ordered[-2])
    predicted_one_hot = [float(adjudication == label) for label in classes]
    guardrail_source = guardrail[2] if guardrail else "none"
    guardrail_one_hot = [
        float(guardrail_source == source) for source in CONFIDENCE_GUARDRAIL_SOURCES
    ]
    page_count = max(1, len(pages))
    document_mix = [
        sum(page.document_type == document_type for page in pages) / page_count
        for document_type in CONFIDENCE_DOCUMENT_TYPES
    ]
    return np.asarray(
        [
            raw_confidence,
            margin,
            disagreement,
            len(record.missing_fields) / 8.0,
            record.evidence_quality,
            record.mean_ocr_confidence / 100.0,
            float(guardrail is not None),
            *[float(value) for value in probabilities],
            *predicted_one_hot,
            *guardrail_one_hot,
            *document_mix,
        ],
        dtype=np.float64,
    )


def review_approval_feature_dict(
    probabilities: np.ndarray,
    record: Extraction,
    pages: list[PageEvidence],
    output: dict[str, str],
    evidence_prediction: str,
    field_confidences: dict[str, float],
    classes: list[str],
) -> dict[str, float]:
    """Identity-free visible-evidence features for a review-only specialist."""
    probability = {
        label: float(probabilities[index]) for index, label in enumerate(classes)
    }
    row: dict[str, float] = {
        "bias": 1.0,
        "p:approved": probability["APPROVED"],
        "p:denied": probability["DENIED"],
        "p:review": probability["NEEDS_REVIEW"],
        "p:approve_minus_deny": (
            probability["APPROVED"] - probability["DENIED"]
        ),
        "p:approve_minus_review": (
            probability["APPROVED"] - probability["NEEDS_REVIEW"]
        ),
        f"evidence:{evidence_prediction}": 1.0,
        f"visa:{output['visa_class']}": 1.0,
        f"fee:{output['fee_status']}": 1.0,
        f"purpose:{output['declared_purpose']}": 1.0,
        f"species:{output['species_code']}": 1.0,
        f"world:{output['home_world']}": 1.0,
        "risk:none": float(output["risk_flags"] == "none"),
        "risk:non_none": float(output["risk_flags"] != "none"),
        "sponsor:missing": float(output["sponsor_id"] == "SPN-0000"),
        "sponsor:revoked": float(output["sponsor_id"] in REVOKED_SPONSORS),
        "missing:count": float(len(record.missing_fields)),
    }
    for field in record.missing_fields:
        row[f"missing:{field}"] = 1.0
    for field, confidence in field_confidences.items():
        row[f"field_confidence:{field}"] = float(confidence)

    page_types = Counter(page.document_type for page in pages)
    for document_type in CONFIDENCE_DOCUMENT_TYPES:
        selected = [
            page for page in pages if page.document_type == document_type
        ]
        row[f"pages:{document_type}"] = float(page_types[document_type])
        if selected:
            row[f"ocr_mean:{document_type}"] = float(
                np.mean(
                    [page.ocr_mean_confidence / 100.0 for page in selected]
                )
            )
            row[f"native_zero:{document_type}"] = float(
                np.mean(
                    [page.visible_native_characters == 0 for page in selected]
                )
            )

    text = normalized("\n".join(page.text for page in pages))
    for marker in (
        "OBSERVED FLAGS NONE",
        "AMOUNT 809 00",
        "DIP WAIVER",
        "FEE STATUS PAID",
        "FEE STATUS WAIVED",
        "REGISTRY NAME",
        "PLANETARY REGISTRY EXTRACT",
        "SPONSOR ATTESTATION",
        "PRIMARY INTAKE",
        "PASSPORT IMAGE",
        "SCAN IMAGE",
        "COPY ARTIFACT",
        "VISA CLASS TORN",
        "FEE STATUS UNKNOWN",
    ):
        row[f"marker:{marker}"] = float(marker in text)
    return row


class DecisionEngine:
    def __init__(self, model_path: str | Path | None = None):
        path = (
            Path(model_path)
            if model_path
            else Path(__file__).parents[1] / "models" / "model.joblib"
        )
        self.bundle = joblib.load(path) if path.exists() else None

    @staticmethod
    def _structured_row(record: Extraction) -> list:
        return [
            record.species_code,
            record.home_world,
            record.visa_class,
            record.sponsor_id,
            record.declared_purpose,
            record.risk_flags,
            record.fee_status,
            int(record.arrival_date.replace("-", ""))
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", record.arrival_date)
            else 19000101,
        ]

    def _model_probabilities(
        self, record: Extraction, pages: list[PageEvidence]
    ) -> tuple[np.ndarray, float]:
        if not self.bundle:
            return np.array([0.34, 0.33, 0.33]), 0.0
        text_model = self.bundle["text_model"]
        structured_model = self.bundle["structured_model"]
        classes = list(self.bundle["classes"])
        text_prob = text_model.predict_proba([visible_model_text(pages)])[0]
        structured_prob = structured_model.predict_proba(
            [self._structured_row(record)]
        )[0]
        text_prob = np.array(
            [text_prob[list(text_model.classes_).index(c)] for c in classes]
        )
        structured_prob = np.array(
            [structured_prob[list(structured_model.classes_).index(c)] for c in classes]
        )
        weight = float(self.bundle.get("text_weight", 0.65))
        probability = weight * text_prob + (1.0 - weight) * structured_prob
        policy_weight = float(
            self.bundle.get("decision_policy", {}).get("policy_weight", 0.0)
        )
        if policy_weight:
            policy_label = evidence_policy(record, pages)[0]
            policy_probability = np.array(
                [float(label == policy_label) for label in classes]
            )
            probability = (
                1.0 - policy_weight
            ) * probability + policy_weight * policy_probability
        disagreement = float(np.abs(text_prob - structured_prob).sum() / 2.0)
        return probability, disagreement

    def correct_record(
        self, record: Extraction, raw_pages: list[dict] | list[PageEvidence]
    ) -> Extraction:
        if not self.bundle or not self.bundle.get("field_models"):
            return record
        pages = [
            page if isinstance(page, PageEvidence) else PageEvidence.from_dict(page)
            for page in raw_pages
        ]
        text_model = self.bundle["text_model"]
        vectorizer = text_model.steps[0][1]
        matrix = vectorizer.transform([visible_text(pages)])
        updates: dict[str, str] = {}
        name_tokens = self.bundle.get("name_tokens", ())
        if record.applicant_name != "unknown" and name_tokens:
            corrected_tokens: list[str] = []
            changed = False
            for token in record.applicant_name.split():
                score, candidate = max(
                    (
                        SequenceMatcher(
                            None, token.casefold(), known.casefold()
                        ).ratio(),
                        known,
                    )
                    for known in name_tokens
                )
                if score >= 0.70:
                    corrected_tokens.append(candidate)
                    changed = changed or candidate.casefold() != token.casefold()
                else:
                    corrected_tokens.append(token)
            if changed:
                updates["applicant_name"] = " ".join(corrected_tokens)
        fallbacks = {
            "species_code": "unknown",
            "home_world": "unknown",
            "visa_class": "unknown",
            "declared_purpose": "unknown",
            "risk_flags": "none",
            "fee_status": "unknown",
        }
        thresholds = {
            "species_code": 0.97,
            "home_world": 0.97,
            "visa_class": 0.97,
            "declared_purpose": 0.97,
            "fee_status": 0.84,
        }
        for field, model in self.bundle["field_models"].items():
            # Risk evidence is asymmetric and outcome-determinative. Organizer
            # guidance explicitly says invisible flags are under-determined;
            # never infer them from correlations learned across packets.
            if field == "risk_flags":
                continue
            probabilities = model.predict_proba(matrix)[0]
            index = int(np.argmax(probabilities))
            value = str(model.classes_[index])
            confidence = float(probabilities[index])
            current = updates.get(field, getattr(record, field))
            missing = current == fallbacks[field]
            if missing and confidence >= max(0.34, thresholds[field] - 0.2):
                updates[field] = value
            elif field in {
                "species_code",
                "home_world",
                "visa_class",
                "declared_purpose",
            }:
                if value != current and confidence >= thresholds[field]:
                    updates[field] = value
            elif value != current and confidence >= thresholds[field]:
                updates[field] = value
        if not updates:
            return record
        updated = replace(record, **updates)
        fallback_by_field = {
            "applicant_name": "unknown",
            "species_code": "unknown",
            "home_world": "unknown",
            "visa_class": "unknown",
            "sponsor_id": "SPN-0000",
            "arrival_date": "1900-01-01",
            "declared_purpose": "unknown",
            "fee_status": "unknown",
        }
        missing_fields = tuple(
            field
            for field, fallback in fallback_by_field.items()
            if getattr(updated, field) == fallback
        )
        return replace(updated, missing_fields=missing_fields)

    def output_fields(
        self, record: Extraction, raw_pages: list[dict] | list[PageEvidence]
    ) -> dict:
        """Apply extraction estimates before the final policy postcondition."""
        output_fallbacks = (
            self.bundle.get("output_fallbacks", {}) if self.bundle else {}
        )
        output = record.output_fields(output_fallbacks)
        pages = [
            page if isinstance(page, PageEvidence) else PageEvidence.from_dict(page)
            for page in raw_pages
        ]
        if (
            not self.bundle
            or not self.bundle.get("field_models")
            or not self.bundle.get("output_field_thresholds")
        ):
            output.update(visible_output_corrections(output, pages))
            return output
        vectorizer = self.bundle["text_model"].steps[0][1]
        matrix = vectorizer.transform([visible_text(pages)])
        for field, threshold in self.bundle["output_field_thresholds"].items():
            if float(threshold) >= 1.0:
                continue
            model = self.bundle["field_models"][field]
            probabilities = model.predict_proba(matrix)[0]
            index = int(np.argmax(probabilities))
            if float(probabilities[index]) >= float(threshold):
                output[field] = str(model.classes_[index])
        output.update(visible_output_corrections(output, pages))
        return output

    def _field_model_confidences(
        self, pages: list[PageEvidence]
    ) -> dict[str, float]:
        if not self.bundle or not self.bundle.get("field_models"):
            return {}
        vectorizer = self.bundle["text_model"].steps[0][1]
        matrix = vectorizer.transform([visible_text(pages)])
        return {
            field: float(np.max(model.predict_proba(matrix)[0]))
            for field, model in self.bundle["field_models"].items()
        }

    def decide(
        self,
        record: Extraction,
        raw_pages: list[dict] | list[PageEvidence],
        emitted_output: dict | None = None,
    ) -> Decision:
        pages = [
            page if isinstance(page, PageEvidence) else PageEvidence.from_dict(page)
            for page in raw_pages
        ]
        guardrail = deterministic_guardrail(record, pages)
        probabilities, disagreement = self._model_probabilities(record, pages)
        classes = list(self.bundle["classes"]) if self.bundle else list(ADJUDICATIONS)
        probability_map = {
            label: float(probabilities[index]) for index, label in enumerate(classes)
        }

        if guardrail:
            adjudication, confidence, source = guardrail
            model_support = probability_map.get(adjudication, 1 / 3)
            confidence = 0.75 * confidence + 0.25 * model_support
        else:
            policy = self.bundle.get("decision_policy") if self.bundle else None
            adjudication = (
                threshold_policy_label(probabilities, classes, policy)
                if policy
                else expected_utility_label(probabilities, classes)
            )
            confidence = probability_map[adjudication]
            source = "cost_sensitive_hybrid"

        output = (
            emitted_output
            if emitted_output is not None
            else self.output_fields(record, pages)
        )
        emitted_guardrail = emitted_policy_guardrail(record, output, pages)
        if emitted_guardrail:
            emitted_label, emitted_confidence, emitted_source = emitted_guardrail
            should_apply = (emitted_label == "DENIED" and adjudication != "DENIED") or (
                emitted_label == "NEEDS_REVIEW" and adjudication == "APPROVED"
            )
            if should_apply:
                adjudication = emitted_label
                model_support = probability_map.get(adjudication, 1 / 3)
                confidence = 0.75 * emitted_confidence + 0.25 * model_support
                source = emitted_source
                guardrail = emitted_guardrail

        approval_guardrail = visible_approval_guardrail(record, output, pages)
        if (
            approval_guardrail
            and adjudication == "NEEDS_REVIEW"
            and guardrail is None
            and emitted_guardrail is None
        ):
            adjudication, confidence, source = approval_guardrail
            model_support = probability_map.get(adjudication, 1 / 3)
            confidence = 0.75 * confidence + 0.25 * model_support
            guardrail = approval_guardrail

        if (
            self.bundle
            and adjudication == "NEEDS_REVIEW"
            and guardrail is None
            and emitted_guardrail is None
            and self.bundle.get("review_approval_model") is not None
            and self.bundle.get("review_approval_vectorizer") is not None
            and self.bundle.get("review_approval_threshold") is not None
        ):
            head_features = review_approval_feature_dict(
                probabilities,
                record,
                pages,
                output,
                evidence_policy(record, pages)[0],
                self._field_model_confidences(pages),
                classes,
            )
            head_matrix = self.bundle["review_approval_vectorizer"].transform(
                [head_features]
            )
            head_model = self.bundle["review_approval_model"]
            approved_index = list(head_model.classes_).index(True)
            approval_probability = float(
                head_model.predict_proba(head_matrix)[0, approved_index]
            )
            if approval_probability >= float(
                self.bundle["review_approval_threshold"]
            ):
                adjudication = "APPROVED"
                confidence = approval_probability
                source = "learned_review_approval"

        if self.bundle and "confidence_model" in self.bundle:
            features = confidence_feature_row(
                probabilities,
                adjudication,
                confidence,
                disagreement,
                record,
                pages,
                guardrail,
                classes,
            ).reshape(1, -1)
            linear_confidence = float(
                self.bundle["confidence_model"].predict_proba(features)[0, 1]
            )
            tree_model = self.bundle.get("confidence_tree_model")
            if tree_model is not None:
                tree_confidence = float(tree_model.predict_proba(features)[0, 1])
                linear_weight = float(
                    self.bundle.get("confidence_blend_weight", 0.5)
                )
                confidence = (
                    linear_weight * linear_confidence
                    + (1.0 - linear_weight) * tree_confidence
                )
            else:
                confidence = linear_confidence
        else:
            confidence *= 1.0 - min(0.22, 0.12 * disagreement)
            confidence *= 0.88 + 0.12 * record.evidence_quality
        confidence = min(0.99, max(0.05, confidence))
        return Decision(
            adjudication=adjudication,
            confidence=round(confidence, 4),
            source=source,
            probabilities=probability_map,
        )
