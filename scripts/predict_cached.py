#!/usr/bin/env python3
"""Developer utility: build predictions from a resumable visible-OCR cache."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution.decision import DecisionEngine
from solution.predict import build_prediction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/model.joblib"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    engine = DecisionEngine(args.model)
    count = 0
    with args.cache.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            if not line.strip():
                continue
            prediction = build_prediction(json.loads(line), engine)
            output.write(json.dumps(prediction, sort_keys=True) + "\n")
            count += 1
    print(f"Wrote {count} cached predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
