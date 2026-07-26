from __future__ import annotations

import argparse
import csv
import json
import os
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import ADJUDICATIONS
from .decision import (
    DecisionEngine,
    deterministic_guardrail,
    threshold_policy_label,
    visible_text,
)
from .fields import Extraction, extract_fields
from .pdf import PageEvidence, safe_extract_pdf

CLASSES = np.array(ADJUDICATIONS)
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


def build_cache(input_dir: Path, cache_path: Path, case_ids: list[str], workers: int) -> list[dict]:
    cached = load_cache(cache_path)
    missing = [case_id for case_id in case_ids if case_id not in cached]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if missing:
        print(f"Extracting {len(missing)} uncached PDFs with {workers} workers", flush=True)
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
        LogisticRegression(C=3.0, max_iter=2500, random_state=8090),
    )


def make_structured_model() -> object:
    categorical = list(range(len(STRUCTURED_FIELDS)))
    date_column = [len(STRUCTURED_FIELDS)]
    preprocess = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("date", StandardScaler(), date_column),
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


def competition_scores(y: np.ndarray, prediction: np.ndarray, confidence: np.ndarray) -> dict:
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


def field_accuracy(labels: list[dict[str, str]], records: list[Extraction]) -> dict[str, float]:
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
    return {
        field: sum(
            getattr(record, field).strip().casefold() == truth[field].strip().casefold()
            for truth, record in zip(labels, records)
        )
        / len(records)
        for field in fields
    }


def fit_field_models(text_model: object, texts: np.ndarray, labels: list[dict[str, str]]):
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
        [PageEvidence.from_dict(page) for page in item["pages"]]
        for item in evidence
    ]
    records = [
        extract_fields(item["case_id"], item["pages"])
        for item in evidence
    ]
    texts = np.array([visible_text(case_pages) for case_pages in pages], dtype=object)
    y = np.array([row["adjudication"] for row in label_rows])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=8090)

    text_model = make_text_model()
    structured_model = make_structured_model()
    text_prob = np.zeros((len(y), len(CLASSES)), dtype=float)
    structured_prob = np.zeros_like(text_prob)
    oof_records: list[Extraction | None] = [None] * len(y)
    print("Generating leakage-safe out-of-fold field and adjudication predictions", flush=True)
    for fold, (train_indices, validation_indices) in enumerate(cv.split(texts, y), start=1):
        fold_text_model = make_text_model()
        fold_text_model.fit(texts[train_indices], y[train_indices])
        train_labels = [label_rows[index] for index in train_indices]
        fold_field_models = fit_field_models(
            fold_text_model, texts[train_indices], train_labels
        )
        engine = correction_engine(fold_text_model, fold_field_models, train_labels)
        corrected_train = [
            engine.correct_record(records[index], pages[index]) for index in train_indices
        ]
        corrected_validation = [
            engine.correct_record(records[index], pages[index]) for index in validation_indices
        ]
        fold_structured_model = make_structured_model()
        fold_structured_model.fit(
            np.array([structured_row(record) for record in corrected_train], dtype=object),
            y[train_indices],
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
        print(f"  completed fold {fold}/5", flush=True)
    corrected_records = [record for record in oof_records if record is not None]
    if len(corrected_records) != len(records):
        raise RuntimeError("Out-of-fold correction did not cover every training case")

    guardrails = [
        deterministic_guardrail(record, case_pages)
        for record, case_pages in zip(corrected_records, pages)
    ]
    best: (
        tuple[
            float,
            float,
            dict[str, float],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
        | None
    ) = None
    for text_weight in np.linspace(0.35, 0.9, 12):
        probability = text_weight * text_prob + (1.0 - text_weight) * structured_prob
        for approve_threshold in np.arange(0.3, 0.96, 0.05):
            for approve_margin in np.arange(0.0, 0.61, 0.05):
                for deny_threshold in np.arange(0.3, 0.71, 0.05):
                    policy = {
                        "approve_threshold": float(approve_threshold),
                        "approve_margin": float(approve_margin),
                        "deny_threshold": float(deny_threshold),
                    }
                    predicted = np.array(
                        [
                            threshold_policy_label(
                                case_probability, list(CLASSES), policy
                            )
                            for case_probability in probability
                        ]
                    )
                    guardrail_mask = np.zeros(len(y), dtype=bool)
                    raw_confidence = np.array(
                        [
                            case_probability[list(CLASSES).index(label)]
                            for case_probability, label in zip(probability, predicted)
                        ]
                    )
                    for index, guardrail in enumerate(guardrails):
                        if guardrail:
                            label, confidence, _ = guardrail
                            predicted[index] = label
                            guardrail_mask[index] = True
                            support = probability[index, list(CLASSES).index(label)]
                            raw_confidence[index] = 0.75 * confidence + 0.25 * support
                    baseline = competition_scores(y, predicted, raw_confidence)
                    catastrophic = baseline["catastrophic_false_approvals"]
                    objective = (
                        baseline["classification_score"]
                        if catastrophic <= 5
                        else baseline["classification_score"] - 100.0 - catastrophic
                    )
                    if best is None or objective > best[0]:
                        best = (
                            objective,
                            float(text_weight),
                            policy,
                            probability,
                            predicted,
                            guardrail_mask,
                        )
    assert best is not None
    _, text_weight, decision_policy, probability, predicted, guardrail_mask = best

    ordered = np.sort(probability, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    disagreement = np.abs(text_prob - structured_prob).sum(axis=1) / 2.0
    raw_confidence = np.array(
        [
            case_probability[list(CLASSES).index(label)]
            for case_probability, label in zip(probability, predicted)
        ]
    )
    for index, (record, case_pages) in enumerate(zip(corrected_records, pages)):
        guardrail = deterministic_guardrail(record, case_pages)
        if guardrail:
            label, confidence, _ = guardrail
            support = probability[index, list(CLASSES).index(label)]
            raw_confidence[index] = 0.75 * confidence + 0.25 * support
    confidence_features = np.column_stack(
        [
            raw_confidence,
            margin,
            disagreement,
            np.array([len(record.missing_fields) for record in corrected_records]),
            np.array([record.evidence_quality for record in corrected_records]),
            np.array(
                [record.mean_ocr_confidence / 100.0 for record in corrected_records]
            ),
            guardrail_mask.astype(float),
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

    report = {
        "cases": len(y),
        "text_weight": text_weight,
        "decision_policy": decision_policy,
        "oof_scores": competition_scores(y, predicted, calibrated_confidence),
        "confusion_labels": list(CLASSES),
        "confusion": confusion_matrix(y, predicted, labels=CLASSES).tolist(),
        "field_accuracy": field_accuracy(label_rows, corrected_records),
    }
    print(json.dumps(report, indent=2), flush=True)

    print("Fitting final models on all training cases", flush=True)
    text_model.fit(texts, y)
    field_models = fit_field_models(text_model, texts, label_rows)
    final_engine = correction_engine(text_model, field_models, label_rows)
    final_records = [
        final_engine.correct_record(record, case_pages)
        for record, case_pages in zip(records, pages)
    ]
    structured_model.fit(
        np.array([structured_row(record) for record in final_records], dtype=object),
        y,
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
    parser.add_argument("--report", type=Path, default=Path("reports/training_report.json"))
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
