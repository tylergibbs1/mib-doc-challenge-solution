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
    parser.add_argument(
        "--allow-failed-states",
        action="store_true",
        help="retain conservative production fallbacks for failed extraction states",
    )
    parser.add_argument(
        "--replacement-states",
        type=Path,
        action="append",
        default=[],
        help="JSONL states that may replace failed rows with the same case ID",
    )
    parser.add_argument(
        "--production-input-dir",
        type=Path,
        help="PDF directory used to reproduce production retry ordering",
    )
    parser.add_argument(
        "--production-retry-limit",
        type=int,
        help="apply replacements only to the first N production retry candidates",
    )
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
    if args.replacement_states:
        failed_case_ids = {
            str(row.get("case_id"))
            for _, row in records
            if row.get("error")
            or (
                row.get("extraction", {}).get("attempts")
                and row["extraction"]["attempts"][-1].get("status") != "success"
            )
        }
        replacements = {}
        for replacement_path in args.replacement_states:
            for line in replacement_path.read_text().splitlines():
                replacement = json.loads(line)
                replacement.pop("_distributed_generation", None)
                case_id = str(replacement.get("case_id"))
                if case_id not in failed_case_ids:
                    raise SystemExit(
                        f"replacement targets non-failed case: {case_id}"
                    )
                replacements[case_id] = replacement
        if args.production_retry_limit is not None:
            if args.production_input_dir is None:
                raise SystemExit(
                    "--production-retry-limit requires --production-input-dir"
                )
            pdfs = sorted(
                (
                    str(path)
                    for path in args.production_input_dir.glob("*.pdf")
                ),
                key=lambda path: (
                    os.path.getsize(path),
                    Path(path).name,
                ),
            )
            ordered_states = [row for _, row in records]
            eligible = {
                str(state.get("case_id"))
                for _, state, _ in predict._retry_candidates(
                    ordered_states, pdfs
                )[: args.production_retry_limit]
            }
            replacements = {
                case_id: replacement
                for case_id, replacement in replacements.items()
                if case_id in eligible
            }
            print(
                "production retry replacement cases="
                + ",".join(sorted(replacements))
            )
        records = [
            (key, replacements.get(str(row.get("case_id")), row))
            for key, row in records
        ]
        print(
            f"applied {len(replacements)} replacement extraction states"
        )
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
    if failed and not args.allow_failed_states:
        raise SystemExit(
            f"{len(failed)} extraction states failed; first={failed[:10]}"
        )
    if failed:
        print(
            f"retaining {len(failed)} conservative extraction fallbacks; "
            f"first={failed[:10]}"
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
