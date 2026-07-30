"""Pure-Python inference for the score-optimal review resolver.

The model is deliberately narrow: it may act only on the deterministic
pipeline's ``insufficient_evidence`` reviews.  Features describe visible
evidence quality and layout; case identifiers and open-vocabulary applicant or
sponsor identities are excluded.  Training lives in ``tools/`` and exports a
small forest as JSON so runtime needs no sklearn dependency.
"""
import json
import os
import struct
from collections import Counter
from datetime import date
from pathlib import Path


FIELDS = (
    "applicant_name", "species_code", "home_world", "visa_class",
    "sponsor_id", "arrival_date", "declared_purpose", "risk_flags",
    "fee_status",
)
CATEGORICAL_VALUES = (
    "species_code", "home_world", "visa_class", "declared_purpose",
    "risk_flags", "fee_status",
)
LABELS = ("APPROVED", "DENIED", "NEEDS_REVIEW")
UTILITY = {
    "APPROVED": {"APPROVED": 8, "DENIED": 0, "NEEDS_REVIEW": 2},
    "DENIED": {"APPROVED": -4, "DENIED": 8, "NEEDS_REVIEW": 2},
    "NEEDS_REVIEW": {"APPROVED": 1, "DENIED": 1, "NEEDS_REVIEW": 8},
}
_MODEL = None


def _f32(value):
    """Match sklearn tree inference's float32 feature conversion."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def primary_reason(detail):
    reasons = detail.get("reasons") or ["<none>"]
    return str(reasons[0]).split(":")[0]


def _candidate_summary(features, state, detail):
    pools = state.get("pools") or {}
    evidence = detail.get("field_evidence") or {}
    sources = detail.get("sources") or {}
    extracted = set(detail.get("extracted_fields") or ())
    for field in FIELDS:
        candidates = pools.get(field) or ()
        features[f"present:{field}"] = int(field in extracted)
        features[f"pool_n:{field}"] = min(len(candidates), 6)
        if field in sources:
            features[f"source:{field}={sources[field]}"] = 1
        if field in evidence:
            rank, score, agreement = evidence[field]
            features[f"rank:{field}"] = float(rank)
            features[f"score:{field}"] = float(score) / 100.0
            features[f"agreement:{field}"] = min(float(agreement), 3.0)
        if candidates:
            scores = [
                float(candidate[3])
                for candidate in candidates
                if len(candidate) > 3
            ]
            if scores:
                features[f"pool_score_max:{field}"] = max(scores) / 100.0
                features[f"pool_score_min:{field}"] = min(scores) / 100.0
            features[f"pool_sources:{field}"] = len({
                str(candidate[1])
                for candidate in candidates
                if len(candidate) > 1
            })


def evidence_features(state, detail, prediction):
    """Return the exact training/runtime feature map."""
    features = {
        f"baseline={prediction['adjudication']}": 1,
        f"reason={primary_reason(detail)}": 1,
        "extracted_n": len(detail.get("extracted_fields") or ()),
        "page_type_n": len(detail.get("page_types") or ()),
        "scan_pages": min(float(state.get("n_scan_pages", 0)), 6.0),
        "mean_ocr_conf": float(state.get("mean_ocr_conf", 0.0)),
        "hq_used": int(bool(state.get("hq_used"))),
        "native_ledger": int("native_ledger" in state),
    }
    for page_type, count in Counter(
            detail.get("page_types") or ()).items():
        features[f"page_type:{page_type}"] = min(count, 4)

    doc_notes = state.get("doc_notes") or {}
    for field in doc_notes.get("absent_fields") or ():
        features[f"absent:{field}"] = 1
    features["absent_n"] = len(doc_notes.get("absent_fields") or ())
    features["registry_embargo"] = int(bool(
        doc_notes.get("registry_embargo")))
    features["watermark_pages"] = min(
        float(doc_notes.get("watermark_pages") or 0), 4.0)
    bio = doc_notes.get("bio_confidence")
    if bio is not None:
        features["bio_confidence"] = float(bio) / 100.0
        features["bio_confidence_present"] = 1

    for field in CATEGORICAL_VALUES:
        features[f"value:{field}={prediction.get(field)}"] = 1

    try:
        arrival = date.fromisoformat(str(prediction["arrival_date"]))
        creation = date.fromisoformat(str(state["pdf_creation_date"]))
        features["arrival_age_days"] = float(
            (creation - arrival).days) / 180
    except (KeyError, TypeError, ValueError):
        pass

    _candidate_summary(features, state, detail)
    return features


def _model_path():
    return Path(__file__).resolve().parents[1] / "models" / \
        "review_resolver.json"


def _load_model():
    global _MODEL
    if _MODEL is None:
        path = _model_path()
        if not path.exists():
            _MODEL = {}
        else:
            payload = json.loads(path.read_text())
            if payload.get("schema") != "mib-review-resolver-v1":
                raise ValueError("unsupported review resolver schema")
            payload["_feature_index"] = {
                name: index
                for index, name in enumerate(payload["feature_names"])
            }
            _MODEL = payload
    return _MODEL


def _tree_probability(tree, values):
    node = 0
    left = tree["left"]
    right = tree["right"]
    while left[node] != -1:
        feature_value = values.get(tree["feature"][node], 0.0)
        node = (
            left[node]
            if feature_value <= tree["threshold"][node]
            else right[node]
        )
    probability = tree["probability"][node]
    total = sum(probability)
    if total <= 0:
        return [1.0 / len(LABELS)] * len(LABELS)
    return [float(value) / total for value in probability]


def resolve(state, detail, prediction):
    """Return an audited model action, or ``None`` outside its narrow scope."""
    # This learned head can replace a manual-required review with a terminal
    # action even when the packet is visibly incomplete. Keep direct library
    # use fail-closed; the submitted run.sh profile enables it explicitly.
    if os.environ.get("MIB_REVIEW_MODEL", "0") != "1":
        return None
    if prediction.get("adjudication") != "NEEDS_REVIEW":
        return None
    if primary_reason(detail) != "insufficient_evidence":
        return None
    model = _load_model()
    if not model:
        return None

    feature_map = evidence_features(state, detail, prediction)
    feature_index = model["_feature_index"]
    values = {
        feature_index[name]: _f32(value)
        for name, value in feature_map.items()
        if name in feature_index
    }
    probability = [0.0] * len(model["classes"])
    for tree in model["trees"]:
        leaf = _tree_probability(tree, values)
        for index, value in enumerate(leaf):
            probability[index] += value
    probability = [
        value / len(model["trees"]) for value in probability
    ]
    class_index = {
        label: index for index, label in enumerate(model["classes"])
    }
    expected = {
        action: sum(
            probability[class_index[truth]] * UTILITY[truth][action]
            for truth in LABELS
        )
        for action in LABELS
    }
    raw_action = max(LABELS, key=lambda candidate: expected[candidate])
    action = raw_action
    try:
        approval_margin = float(
            os.environ.get("MIB_REVIEW_MARGIN", "0"))
    except ValueError:
        approval_margin = 0.0
    margin = expected[raw_action] - expected["NEEDS_REVIEW"]
    if raw_action == "APPROVED" and margin < approval_margin:
        action = "NEEDS_REVIEW"
    action_probability = probability[class_index[action]]
    return {
        "action": action,
        "raw_action": raw_action,
        "approval_margin": margin,
        # The forest's direct action probability was better calibrated on the
        # frozen calibration split than the pre-registered 50% shrink.
        "confidence": min(0.99, max(0.02, action_probability)),
        "probabilities": {
            label: probability[class_index[label]] for label in LABELS
        },
        "expected_utility": expected,
        "model_sha256": model.get("training", {}).get("payload_sha256"),
    }
