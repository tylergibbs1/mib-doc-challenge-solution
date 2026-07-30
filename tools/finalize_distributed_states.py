#!/usr/bin/env python3
"""Apply the unchanged global batch decision pass to distributed states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MIB_NATIVE_SCAN_OCR", "1")
os.environ.setdefault("MIB_REVIEW_MODEL", "1")

from scripts import predict  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    records = []
    source_files = sorted(args.states_dir.glob("**/states.jsonl"))
    if len(source_files) != 16:
        raise SystemExit(f"expected 16 state files, found {len(source_files)}")
    for source in source_files:
        for line in source.read_text().splitlines():
            row = json.loads(line)
            generation = row.pop("_distributed_generation")
            key = (
                int(generation["worker"]),
                int(generation["segment"]),
                int(generation["position"]),
            )
            records.append((key, row))
    records.sort(key=lambda item: item[0])
    states = [row for _, row in records]
    if len(states) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} states, found {len(states)}"
        )
    case_ids = [str(state.get("case_id")) for state in states]
    if len(set(case_ids)) != len(case_ids):
        raise SystemExit("duplicate case ID in distributed states")
    failed = [
        state["case_id"]
        for state in states
        if state.get("error")
        or (
            state.get("extraction", {}).get("attempts")
            and state["extraction"]["attempts"][-1].get("status") != "success"
        )
    ]
    if failed:
        raise SystemExit(
            f"{len(failed)} extraction states failed; first={failed[:10]}"
        )

    epoch, batch_revoked, _, _, _ = predict._batch_decision_inputs(states)
    predict._write_predictions(
        states, epoch, args.output_path, batch_revoked
    )
    actual = sha256(args.output_path)
    if actual != args.expected_sha256:
        raise SystemExit(
            f"prediction hash mismatch: {actual} != {args.expected_sha256}"
        )
    print(
        f"wrote {len(states)} predictions with verified sha256={actual}"
    )


if __name__ == "__main__":
    main()
