from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np

from .constants import (
    ADJUDICATIONS,
    DISQUALIFYING_FLAGS,
    REVIEW_FLAGS,
    REVOKED_SPONSORS,
)
from .fields import Extraction, normalized
from .pdf import PageEvidence


@dataclass
class Decision:
    adjudication: str
    confidence: float
    source: str
    probabilities: dict[str, float]


def visible_text(pages: list[PageEvidence]) -> str:
    # Case identifiers are output keys, not evidence. Removing them prevents
    # character n-grams from learning accidental train-only identity features.
    return "\n\n".join(
        f"[PAGE {page.page_number} {page.document_type.upper()}]\n"
        + re.sub(
            r"\bMIB[-\s]\d{6}\b",
            "MIB-CASE",
            page.text,
            flags=re.IGNORECASE,
        )
        for page in pages
    )


def explicit_manual_finding(pages: list[PageEvidence]) -> tuple[str, float]:
    candidates: list[tuple[int, float, str]] = []
    for page in pages:
        if page.document_type != "manual":
            continue
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
    if len(flags & REVIEW_FLAGS) >= 2 and "RESCINDED DENIAL" not in text:
        return "DENIED", 0.82, "combined_review_flags"
    return None


def expected_utility_label(probabilities: np.ndarray, classes: list[str]) -> str:
    probability = {label: float(probabilities[index]) for index, label in enumerate(classes)}
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
    probability = {label: float(probabilities[index]) for index, label in enumerate(classes)}
    if (
        probability["APPROVED"] >= policy["approve_threshold"]
        and probability["APPROVED"] - probability["DENIED"]
        >= policy["approve_margin"]
    ):
        return "APPROVED"
    if probability["DENIED"] >= policy["deny_threshold"]:
        return "DENIED"
    return "NEEDS_REVIEW"


class DecisionEngine:
    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path) if model_path else Path(__file__).parents[1] / "models" / "model.joblib"
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
        text_prob = text_model.predict_proba([visible_text(pages)])[0]
        structured_prob = structured_model.predict_proba([self._structured_row(record)])[0]
        text_prob = np.array([text_prob[list(text_model.classes_).index(c)] for c in classes])
        structured_prob = np.array(
            [structured_prob[list(structured_model.classes_).index(c)] for c in classes]
        )
        weight = float(self.bundle.get("text_weight", 0.65))
        probability = weight * text_prob + (1.0 - weight) * structured_prob
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
                        SequenceMatcher(None, token.casefold(), known.casefold()).ratio(),
                        known,
                    )
                    for known in name_tokens
                )
                if score >= 0.82:
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
                continue
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

    def decide(self, record: Extraction, raw_pages: list[dict] | list[PageEvidence]) -> Decision:
        pages = [
            page if isinstance(page, PageEvidence) else PageEvidence.from_dict(page)
            for page in raw_pages
        ]
        guardrail = deterministic_guardrail(record, pages)
        probabilities, disagreement = self._model_probabilities(record, pages)
        classes = list(self.bundle["classes"]) if self.bundle else list(ADJUDICATIONS)
        probability_map = {label: float(probabilities[index]) for index, label in enumerate(classes)}

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

        ordered = np.sort(probabilities)
        margin = float(ordered[-1] - ordered[-2])
        if self.bundle and "confidence_model" in self.bundle:
            features = [[
                confidence,
                margin,
                disagreement,
                len(record.missing_fields),
                record.evidence_quality,
                record.mean_ocr_confidence / 100.0,
                1.0 if guardrail else 0.0,
            ]]
            confidence = float(self.bundle["confidence_model"].predict_proba(features)[0, 1])
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
