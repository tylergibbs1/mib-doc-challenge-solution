#!/usr/bin/env python3
"""Build or resume a visible-evidence OCR cache without training a model."""

import argparse
import csv
import os
import sys
from pathlib import Path

# Prevent each worker's OCR and numerical libraries from starting nested
# thread pools. This must happen before importing the ML/OCR modules.
for variable in (
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution.train import build_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    # This developer-only cache builder may use additional host cores. The
    # submitted runtime remains capped at four workers in solution.predict.
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        case_ids = [row["case_id"] for row in csv.DictReader(handle)]
    evidence = build_cache(
        args.input_dir,
        args.cache,
        case_ids,
        max(1, min(10, args.workers)),
    )
    print(f"Cached visible evidence for {len(evidence)} cases at {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
