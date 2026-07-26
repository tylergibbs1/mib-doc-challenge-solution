from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .decision import DecisionEngine
from .fields import extract_fields
from .pdf import safe_extract_pdf


def _extract(path: Path) -> dict:
    return safe_extract_pdf(path)


def build_prediction(evidence: dict, engine: DecisionEngine) -> dict:
    record = extract_fields(evidence["case_id"], evidence["pages"])
    record = engine.correct_record(record, evidence["pages"])
    decision = engine.decide(record, evidence["pages"])
    prediction = record.output_fields()
    prediction["adjudication"] = decision.adjudication
    prediction["confidence"] = decision.confidence
    return prediction


def predict_directory(input_dir: Path, output_path: Path, workers: int) -> int:
    pdfs = sorted(input_dir.glob("*.pdf"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine = DecisionEngine()
    completed = 0
    with (
        output_path.open("w", encoding="utf-8") as output,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        for evidence in pool.map(_extract, pdfs, chunksize=1):
            prediction = build_prediction(evidence, engine)
            output.write(json.dumps(prediction, sort_keys=True) + "\n")
            output.flush()
            completed += 1
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    count = predict_directory(args.input_dir, args.output_path, max(1, min(4, args.workers)))
    print(f"Wrote {count} predictions to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
