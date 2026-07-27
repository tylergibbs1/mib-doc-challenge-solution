from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import ADJUDICATIONS
from .decision import (
    DecisionEngine,
    confidence_feature_row,
    deterministic_guardrail,
    emitted_policy_guardrail,
    evidence_policy,
    threshold_policy_label,
    visible_model_text,
    visible_text,
)
from .fields import Extraction, extract_fields, visible_output_corrections
from .pdf import PageEvidence, safe_extract_pdf

CLASSES = np.array(ADJUDICATIONS)
FIELD_WEIGHTS = {
    "applicant_name": 5,
    "species_code": 6,
    "home_world": 5,
    "visa_class": 5,
    "sponsor_id": 5,
    "arrival_date": 4,
    "declared_purpose": 3,
    "risk_flags": 8,
    "fee_status": 4,
}
STRUCTURED_FIELDS = (
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "declared_purpose",
    "risk_flags",
    "fee_status",
)
FIELD_MODEL_FIELDS = (
    "species_code",
    "home_world",
    "visa_class",
    "declared_purpose",
    "fee_status",
)
OUTPUT_PRIOR_FIELDS = (
    "species_code",
    "home_world",
    "visa_class",
    "arrival_date",
    "declared_purpose",
    "fee_status",
)


def _extract(path: Path) -> dict:
    return safe_extract_pdf(path)


def read_labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def load_cache(path: Path) -> dict[str, dict]:
    cached: dict[str, dict] = {}
    if not path.exists():
        return cached
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                cached[value["case_id"]] = value
    return cached


def build_cache(
    input_dir: Path, cache_path: Path, case_ids: list[str], workers: int
) -> list[dict]:
    cached = load_cache(cache_path)
    missing = [case_id for case_id in case_ids if case_id not in cached]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if missing:
        print(
            f"Extracting {len(missing)} uncached PDFs with {workers} workers",
            flush=True,
        )
        paths = [input_dir / f"{case_id}.pdf" for case_id in missing]
        with (
            cache_path.open("a", encoding="utf-8") as output,
            ProcessPoolExecutor(max_workers=workers) as pool,
        ):
            for index, evidence in enumerate(
                pool.map(_extract, paths, chunksize=1), start=1
            ):
                output.write(json.dumps(evidence, sort_keys=True) + "\n")
                output.flush()
                cached[evidence["case_id"]] = evidence
                if index % 25 == 0 or index == len(paths):
                    print(f"  extracted {index}/{len(paths)}", flush=True)
    return [cached[case_id] for case_id in case_ids]


def structured_row(record: Extraction) -> list:
    date_number = (
        int(record.arrival_date.replace("-", ""))
        if record.arrival_date[:2] == "20"
        else 19000101
    )
    return [getattr(record, field) for field in STRUCTURED_FIELDS] + [date_number]


def make_text_model() -> object:
    return make_pipeline(
        FeatureUnion(
            [
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_df=0.995,
                        max_features=65_000,
                        lowercase=True,
                        strip_accents="unicode",
                        sublinear_tf=True,
                        dtype=np.float32,
                    ),
                ),
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.995,
                        max_features=35_000,
                        lowercase=True,
                        strip_accents="unicode",
                        sublinear_tf=True,
                        dtype=np.float32,
                    ),
                ),
            ],
            transformer_weights={"char": 1.0, "word": 0.7},
        ),
        LogisticRegression(C=3.0, max_iter=2500, random_state=8090),
    )


def make_structured_model() -> object:
    categorical = list(range(len(STRUCTURED_FIELDS)))
    date_column = [len(STRUCTURED_FIELDS)]
    preprocess = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            (
                "date",
                StandardScaler(),
                date_column,
            ),
        ]
    )
    return make_pipeline(
        preprocess,
        ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.8,
            class_weight="balanced",
            # Prediction is called once per packet while OCR is already
            # process-parallel. Avoid nested joblib pools and warning floods.
            n_jobs=1,
            random_state=8090,
        ),
    )


def aligned_probabilities(model: object, x, classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    return np.column_stack(
        [raw[:, list(model.classes_).index(label)] for label in classes]
    )


def competition_scores(
    y: np.ndarray, prediction: np.ndarray, confidence: np.ndarray
) -> dict:
    correct = prediction == y
    catastrophic = (y == "DENIED") & (prediction == "APPROVED")
    conservative = (prediction == "NEEDS_REVIEW") & ~correct
    missed_review = (y == "NEEDS_REVIEW") & ~correct
    other_wrong = ~correct & ~catastrophic & ~conservative & ~missed_review
    raw = (
        8 * correct.sum()
        - 4 * catastrophic.sum()
        + 2 * conservative.sum()
        + missed_review.sum()
    )
    classification = 80.0 * raw / (8.0 * len(y))
    brier = float(np.mean((confidence - correct.astype(float)) ** 2))
    calibration = 20.0 * max(0.0, 1.0 - 2.0 * brier)
    return {
        "accuracy": float(correct.mean()),
        "classification_score": float(classification),
        "calibration_score": float(calibration),
        "combined_score": float(classification + calibration),
        "mean_brier": brier,
        "catastrophic_false_approvals": int(catastrophic.sum()),
        "conservative_reviews": int(conservative.sum()),
        "missed_reviews": int(missed_review.sum()),
        "other_wrong": int(other_wrong.sum()),
    }


def apply_policy_constraints(
    probability: np.ndarray,
    initial_prediction: np.ndarray,
    guardrails: list[tuple[str, float, str] | None],
    emitted_guardrails: list[tuple[str, float, str] | None],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, float, str] | None]]:
    """Apply training-time constraints in the same order as inference."""
    predicted = initial_prediction.copy()
    raw_confidence = np.array(
        [
            case_probability[list(CLASSES).index(label)]
            for case_probability, label in zip(probability, predicted)
        ]
    )
    applied_guardrails: list[tuple[str, float, str] | None] = [None] * len(predicted)
    for index, guardrail in enumerate(guardrails):
        if not guardrail:
            continue
        label, confidence, _ = guardrail
        predicted[index] = label
        support = probability[index, list(CLASSES).index(label)]
        raw_confidence[index] = 0.75 * confidence + 0.25 * support
        applied_guardrails[index] = guardrail
    for index, guardrail in enumerate(emitted_guardrails):
        if not guardrail:
            continue
        label, confidence, _ = guardrail
        should_apply = (label == "DENIED" and predicted[index] != "DENIED") or (
            label == "NEEDS_REVIEW" and predicted[index] == "APPROVED"
        )
        if not should_apply:
            continue
        predicted[index] = label
        support = probability[index, list(CLASSES).index(label)]
        raw_confidence[index] = 0.75 * confidence + 0.25 * support
        applied_guardrails[index] = guardrail
    return predicted, raw_confidence, applied_guardrails


def field_accuracy(
    labels: list[dict[str, str]],
    records: list[Extraction],
    output_fallbacks: list[dict[str, str]] | None = None,
    output_overrides: list[dict[str, str]] | None = None,
) -> dict[str, float]:
    fields = (
        "applicant_name",
        "species_code",
        "home_world",
        "visa_class",
        "sponsor_id",
        "arrival_date",
        "declared_purpose",
        "risk_flags",
        "fee_status",
    )
    if output_fallbacks is None:
        output_fallbacks = [{} for _ in records]
    if output_overrides is None:
        output_overrides = [{} for _ in records]
    output_rows = [
        record.output_fields(fallbacks) | overrides
        for record, fallbacks, overrides in zip(
            records, output_fallbacks, output_overrides
        )
    ]
    return {
        field: sum(
            output[field].strip().casefold() == truth[field].strip().casefold()
            for truth, output in zip(labels, output_rows)
        )
        / len(records)
        for field in fields
    }


def competition_extraction_score(
    labels: list[dict[str, str]],
    records: list[Extraction],
    output_fallbacks: list[dict[str, str]] | None = None,
    output_overrides: list[dict[str, str]] | None = None,
) -> float:
    earned = 0
    available = 0
    if output_fallbacks is None:
        output_fallbacks = [{} for _ in records]
    if output_overrides is None:
        output_overrides = [{} for _ in records]
    for truth, record, fallbacks, overrides in zip(
        labels, records, output_fallbacks, output_overrides
    ):
        output = record.output_fields(fallbacks) | overrides
        unrecoverable = {
            field.strip()
            for field in truth.get("unrecoverable_fields", "").split("|")
            if field.strip()
        }
        for field, weight in FIELD_WEIGHTS.items():
            if field in unrecoverable:
                continue
            available += weight
            predicted = output[field].strip().casefold()
            expected = truth[field].strip().casefold()
            if field == "risk_flags":
                predicted = "|".join(sorted(predicted.split("|")))
                expected = "|".join(sorted(expected.split("|")))
            if predicted == expected:
                earned += weight
    return 50.0 * earned / available


def learn_output_fallbacks(labels: list[dict[str, str]]) -> dict[str, str]:
    """Learn identity-free categorical priors for unresolved output fields."""
    fallbacks: dict[str, str] = {}
    for field in OUTPUT_PRIOR_FIELDS:
        counts = Counter(row[field] for row in labels)
        fallbacks[field] = min(
            counts,
            key=lambda value: (-counts[value], value),
        )
    return fallbacks


def fit_field_models(
    text_model: object, texts: np.ndarray, labels: list[dict[str, str]]
):
    vectorizer = text_model.steps[0][1]
    matrix = vectorizer.transform(texts)
    models = {}
    for field in FIELD_MODEL_FIELDS:
        target = np.array([row[field] for row in labels])
        model = LogisticRegression(C=3.0, max_iter=2500, random_state=8090)
        model.fit(matrix, target)
        models[field] = model
    return models


def correction_engine(
    text_model: object,
    field_models: dict[str, object],
    labels: list[dict[str, str]],
) -> DecisionEngine:
    engine = object.__new__(DecisionEngine)
    engine.bundle = {
        "text_model": text_model,
        "field_models": field_models,
        "name_tokens": sorted(
            {token for row in labels for token in row["applicant_name"].split()}
        ),
    }
    return engine


def train(
    evidence: list[dict],
    label_rows: list[dict[str, str]],
    model_path: Path,
    report_path: Path,
) -> dict:
    pages = [
        [PageEvidence.from_dict(page) for page in item["pages"]] for item in evidence
    ]
    model_pages = [
        [PageEvidence.from_dict(page) for page in item.get("model_pages", ())]
        for item in evidence
    ]
    records = [extract_fields(item["case_id"], item["pages"]) for item in evidence]
    texts = np.array(
        [
            visible_text(explicit_pages)
            if explicit_pages
            else visible_model_text(case_pages)
            for explicit_pages, case_pages in zip(model_pages, pages)
        ],
        dtype=object,
    )
    field_texts = np.array(
        [visible_text(case_pages) for case_pages in pages], dtype=object
    )
    y = np.array([row["adjudication"] for row in label_rows])
    raw_guardrails = [
        deterministic_guardrail(record, case_pages)
        for record, case_pages in zip(records, pages)
    ]
    raw_guardrail_weight = np.array(
        [
            0.25
            if guardrail and guardrail[2] != "embargoed_home_world"
            else 1.0
            for guardrail in raw_guardrails
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=8090)

    text_model = make_text_model()
    structured_model = make_structured_model()
    text_prob = np.zeros((len(y), len(CLASSES)), dtype=float)
    structured_prob = np.zeros_like(text_prob)
    oof_records: list[Extraction | None] = [None] * len(y)
    oof_output_fallbacks: list[dict[str, str] | None] = [None] * len(y)
    field_oof_values = {
        field: np.empty(len(y), dtype=object) for field in FIELD_MODEL_FIELDS
    }
    field_oof_confidence = {
        field: np.zeros(len(y), dtype=float) for field in FIELD_MODEL_FIELDS
    }
    print(
        "Generating leakage-safe out-of-fold field and adjudication predictions",
        flush=True,
    )
    for fold, (train_indices, validation_indices) in enumerate(
        cv.split(texts, y), start=1
    ):
        fold_text_model = make_text_model()
        fold_text_model.fit(
            texts[train_indices],
            y[train_indices],
            logisticregression__sample_weight=raw_guardrail_weight[train_indices],
        )
        train_labels = [label_rows[index] for index in train_indices]
        fold_output_fallbacks = learn_output_fallbacks(train_labels)
        fold_field_models = fit_field_models(
            fold_text_model, field_texts[train_indices], train_labels
        )
        validation_matrix = fold_text_model.steps[0][1].transform(
            field_texts[validation_indices]
        )
        for field, model in fold_field_models.items():
            field_probability = model.predict_proba(validation_matrix)
            best_indices = np.argmax(field_probability, axis=1)
            field_oof_values[field][validation_indices] = model.classes_[best_indices]
            field_oof_confidence[field][validation_indices] = field_probability[
                np.arange(len(validation_indices)), best_indices
            ]
        engine = correction_engine(fold_text_model, fold_field_models, train_labels)
        corrected_train = [
            engine.correct_record(records[index], pages[index])
            for index in train_indices
        ]
        corrected_validation = [
            engine.correct_record(records[index], pages[index])
            for index in validation_indices
        ]
        fold_structured_model = make_structured_model()
        fold_structured_model.fit(
            np.array(
                [structured_row(record) for record in corrected_train], dtype=object
            ),
            y[train_indices],
            extratreesclassifier__sample_weight=np.array(
                [
                    0.25
                    if (
                        (guardrail := deterministic_guardrail(record, pages[index]))
                        and guardrail[2] != "embargoed_home_world"
                    )
                    else 1.0
                    for record, index in zip(corrected_train, train_indices)
                ]
            ),
        )
        text_raw = fold_text_model.predict_proba(texts[validation_indices])
        text_prob[validation_indices] = np.column_stack(
            [
                text_raw[:, list(fold_text_model.classes_).index(label)]
                for label in CLASSES
            ]
        )
        structured_raw = fold_structured_model.predict_proba(
            np.array(
                [structured_row(record) for record in corrected_validation],
                dtype=object,
            )
        )
        structured_prob[validation_indices] = np.column_stack(
            [
                structured_raw[:, list(fold_structured_model.classes_).index(label)]
                for label in CLASSES
            ]
        )
        for index, record in zip(validation_indices, corrected_validation):
            oof_records[index] = record
            oof_output_fallbacks[index] = fold_output_fallbacks
        print(f"  completed fold {fold}/5", flush=True)
    corrected_records = [record for record in oof_records if record is not None]
    if len(corrected_records) != len(records):
        raise RuntimeError("Out-of-fold correction did not cover every training case")
    output_fallbacks = [
        fallbacks for fallbacks in oof_output_fallbacks if fallbacks is not None
    ]
    if len(output_fallbacks) != len(records):
        raise RuntimeError(
            "Out-of-fold output priors did not cover every training case"
        )

    guardrails = [
        deterministic_guardrail(record, case_pages)
        for record, case_pages in zip(corrected_records, pages)
    ]
    policy_predictions = np.array(
        [
            evidence_policy(record, case_pages)[0]
            for record, case_pages in zip(corrected_records, pages)
        ]
    )
    policy_probability = np.column_stack(
        [(policy_predictions == label).astype(float) for label in CLASSES]
    )
    field_model_diagnostics = {}
    output_field_thresholds = {}
    output_overrides: list[dict[str, str]] = [{} for _ in corrected_records]
    for field in FIELD_MODEL_FIELDS:
        truth_values = np.array([row[field] for row in label_rows], dtype=object)
        raw_values = np.array(
            [getattr(record, field) for record in records], dtype=object
        )
        recoverable = np.array(
            [
                field
                not in {
                    value.strip()
                    for value in row.get("unrecoverable_fields", "").split("|")
                    if value.strip()
                }
                for row in label_rows
            ]
        )
        best_threshold = 1.0
        best_accuracy = float(
            (raw_values[recoverable] == truth_values[recoverable]).mean()
        )
        for threshold in np.linspace(0.3, 0.99, 70):
            corrected = np.where(
                field_oof_confidence[field] >= threshold,
                field_oof_values[field],
                raw_values,
            )
            accuracy = float(
                (corrected[recoverable] == truth_values[recoverable]).mean()
            )
            if accuracy > best_accuracy or (
                accuracy == best_accuracy and threshold > best_threshold
            ):
                best_accuracy = accuracy
                best_threshold = float(threshold)
        field_model_diagnostics[field] = {
            "raw_recoverable_accuracy": float(
                (raw_values[recoverable] == truth_values[recoverable]).mean()
            ),
            "model_recoverable_accuracy": float(
                (
                    field_oof_values[field][recoverable] == truth_values[recoverable]
                ).mean()
            ),
            "best_replacement_threshold": best_threshold,
            "best_recoverable_accuracy": best_accuracy,
        }
        emitted_values = np.array(
            [
                record.output_fields(fallbacks)[field]
                for record, fallbacks in zip(corrected_records, output_fallbacks)
            ],
            dtype=object,
        )
        output_threshold = 1.0
        output_accuracy = float((emitted_values == truth_values).mean())
        for threshold in np.linspace(0.3, 0.99, 70):
            output_values = np.where(
                field_oof_confidence[field] >= threshold,
                field_oof_values[field],
                emitted_values,
            )
            accuracy = float((output_values == truth_values).mean())
            if accuracy > output_accuracy or (
                accuracy == output_accuracy and threshold > output_threshold
            ):
                output_accuracy = accuracy
                output_threshold = float(threshold)
        output_field_thresholds[field] = output_threshold
        field_model_diagnostics[field].update(
            {
                "emitted_accuracy_before_output_model": float(
                    (emitted_values == truth_values).mean()
                ),
                "output_replacement_threshold": output_threshold,
                "emitted_accuracy_after_output_model": output_accuracy,
            }
        )
        replace_mask = (
            field_oof_confidence[field] >= output_threshold
            if output_threshold < 1.0
            else np.zeros(len(corrected_records), dtype=bool)
        )
        for index in np.flatnonzero(replace_mask):
            output_overrides[index][field] = str(field_oof_values[field][index])
    for index, (record, fallbacks, case_pages) in enumerate(
        zip(corrected_records, output_fallbacks, pages)
    ):
        output = record.output_fields(fallbacks) | output_overrides[index]
        output_overrides[index].update(
            visible_output_corrections(output, case_pages)
        )
    emitted_outputs = [
        record.output_fields(fallbacks) | overrides
        for record, fallbacks, overrides in zip(
            corrected_records, output_fallbacks, output_overrides
        )
    ]
    emitted_guardrails = [
        emitted_policy_guardrail(record, output, case_pages)
        for record, output, case_pages in zip(corrected_records, emitted_outputs, pages)
    ]
    best: tuple | None = None
    best_unconstrained = None
    for text_weight in np.linspace(0.35, 0.9, 12):
        base_probability = (
            text_weight * text_prob + (1.0 - text_weight) * structured_prob
        )
        for policy_weight in np.arange(0.0, 0.26, 0.05):
            probability = (
                1.0 - policy_weight
            ) * base_probability + policy_weight * policy_probability
            for approve_threshold in np.arange(0.4, 0.71, 0.05):
                for approve_margin in np.arange(0.0, 0.41, 0.05):
                    for deny_threshold in np.arange(0.45, 0.66, 0.05):
                        policy = {
                            "approve_threshold": float(approve_threshold),
                            "approve_margin": float(approve_margin),
                            "deny_threshold": float(deny_threshold),
                            "policy_weight": float(policy_weight),
                        }
                        initial_prediction = np.array(
                            [
                                threshold_policy_label(
                                    case_probability, list(CLASSES), policy
                                )
                                for case_probability in probability
                            ]
                        )
                        (
                            predicted,
                            raw_confidence,
                            applied_guardrails,
                        ) = apply_policy_constraints(
                            probability,
                            initial_prediction,
                            guardrails,
                            emitted_guardrails,
                        )
                        baseline = competition_scores(y, predicted, raw_confidence)
                        catastrophic = baseline["catastrophic_false_approvals"]
                        candidate = (
                            baseline["classification_score"],
                            float(text_weight),
                            policy,
                            probability,
                            predicted,
                            raw_confidence,
                            applied_guardrails,
                            baseline,
                        )
                        if (
                            best_unconstrained is None
                            or candidate[0] > best_unconstrained[0]
                        ):
                            best_unconstrained = candidate
                        if catastrophic <= 5 and (
                            best is None or candidate[0] > best[0]
                        ):
                            best = candidate
    assert best is not None
    assert best_unconstrained is not None
    (
        _,
        text_weight,
        decision_policy,
        probability,
        predicted,
        raw_confidence,
        applied_guardrails,
        _,
    ) = best

    disagreement = np.abs(text_prob - structured_prob).sum(axis=1) / 2.0
    confidence_features = np.vstack(
        [
            confidence_feature_row(
                probability[index],
                str(predicted[index]),
                float(raw_confidence[index]),
                float(disagreement[index]),
                record,
                pages[index],
                applied_guardrails[index],
                list(CLASSES),
            )
            for index, record in enumerate(corrected_records)
        ]
    )
    correctness = (predicted == y).astype(int)
    confidence_model = LogisticRegression(C=0.5, max_iter=1000, random_state=8090)
    if len(np.unique(correctness)) == 2:
        calibration_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=9080)
        calibrated_confidence = cross_val_predict(
            confidence_model,
            confidence_features,
            correctness,
            cv=calibration_cv,
            method="predict_proba",
        )[:, 1]
        confidence_model.fit(confidence_features, correctness)
    else:
        calibrated_confidence = raw_confidence
        confidence_model.fit(
            np.vstack([confidence_features, confidence_features[:1]]),
            np.concatenate([correctness, 1 - correctness[:1]]),
        )

    oof_error_diagnostics = [
        {
            "case_id": label_rows[index]["case_id"],
            "truth": str(y[index]),
            "prediction": str(predicted[index]),
            "confidence": float(calibrated_confidence[index]),
            "probabilities": {
                label: float(probability[index, class_index])
                for class_index, label in enumerate(CLASSES)
            },
            "guardrail": (
                list(applied_guardrails[index])
                if applied_guardrails[index] is not None
                else None
            ),
            "missing_fields": list(corrected_records[index].missing_fields),
            "output": emitted_outputs[index],
        }
        for index in np.flatnonzero(predicted != y)
    ]
    report = {
        "cases": len(y),
        "model_strategy": "word_char_structured_blend",
        "guardrail_training_weight": 0.25,
        "text_weight": text_weight,
        "decision_policy": decision_policy,
        "oof_scores": competition_scores(y, predicted, calibrated_confidence),
        "confusion_labels": list(CLASSES),
        "confusion": confusion_matrix(y, predicted, labels=CLASSES).tolist(),
        "field_accuracy": field_accuracy(label_rows, corrected_records),
        "emitted_field_accuracy": field_accuracy(
            label_rows, corrected_records, output_fallbacks, output_overrides
        ),
        "field_model_oof_diagnostics": field_model_diagnostics,
        "oof_extraction_score": competition_extraction_score(
            label_rows, corrected_records, output_fallbacks, output_overrides
        ),
        "oof_internal_extraction_score": competition_extraction_score(
            label_rows, corrected_records
        ),
        "oof_error_diagnostics": oof_error_diagnostics,
        "evidence_policy_oof_scores": competition_scores(
            y, policy_predictions, np.full(len(y), 0.8)
        ),
        "unconstrained_policy": {
            "model_strategy": "word_char_structured_blend",
            "text_weight": best_unconstrained[1],
            "decision_policy": best_unconstrained[2],
            "scores_before_calibration": best_unconstrained[7],
        },
    }
    print(json.dumps(report, indent=2), flush=True)

    print("Fitting final models on all training cases", flush=True)
    text_model.fit(
        texts,
        y,
        logisticregression__sample_weight=raw_guardrail_weight,
    )
    field_models = fit_field_models(text_model, field_texts, label_rows)
    final_engine = correction_engine(text_model, field_models, label_rows)
    final_records = [
        final_engine.correct_record(record, case_pages)
        for record, case_pages in zip(records, pages)
    ]
    structured_model.fit(
        np.array([structured_row(record) for record in final_records], dtype=object),
        y,
        extratreesclassifier__sample_weight=np.array(
            [
                0.25
                if (
                    (guardrail := deterministic_guardrail(record, case_pages))
                    and guardrail[2] != "embargoed_home_world"
                )
                else 1.0
                for record, case_pages in zip(final_records, pages)
            ]
        ),
    )
    bundle = {
        "version": 2,
        "classes": CLASSES,
        "text_weight": text_weight,
        "decision_policy": decision_policy,
        "text_model": text_model,
        "structured_model": structured_model,
        "field_models": field_models,
        "confidence_model": confidence_model,
        "output_fallbacks": learn_output_fallbacks(label_rows),
        "output_field_thresholds": output_field_thresholds,
        "name_tokens": sorted(
            {token for row in label_rows for token in row["applicant_name"].split()}
        ),
        "training_report": report,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {model_path} ({model_path.stat().st_size / 1_000_000:.1f} MB)")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument(
        "--cache", type=Path, default=Path("training_cache/train_visible_text.jsonl")
    )
    parser.add_argument("--model", type=Path, default=Path("models/model.joblib"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/training_report.json")
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    labels_by_id = read_labels(args.labels)
    case_ids = list(labels_by_id)
    if args.limit:
        case_ids = case_ids[: args.limit]
    evidence = build_cache(
        args.input_dir,
        args.cache,
        case_ids,
        max(1, min(4, args.workers)),
    )
    labels = [labels_by_id[case_id] for case_id in case_ids]
    train(evidence, labels, args.model, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
