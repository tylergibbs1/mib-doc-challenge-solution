#!/usr/bin/env python3
"""Run one sequential segment of the production extraction stream."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MIB_NATIVE_SCAN_OCR", "1")
os.environ.setdefault("MIB_REVIEW_MODEL", "1")

from scripts import predict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--segments-per-worker", type=int, default=4)
    args = parser.parse_args()

    worker, segment = divmod(args.task, args.segments_per_worker)
    pdfs = sorted(
        (str(path) for path in args.input_dir.glob("*.pdf")),
        key=lambda path: (os.path.getsize(path), Path(path).name),
    )
    if not pdfs:
        raise SystemExit("state segment has no PDFs")

    temporary = Path(tempfile.mkdtemp(prefix="mib-segment-", dir="/tmp"))
    shard = predict.Shard(temporary, 0, pdfs)
    while not shard.finished:
        time.sleep(predict.POLL_SECS)
        shard.tick()
    states = predict._collect_states([shard], pdfs, complete=True)
    if len(states) != len(pdfs):
        raise SystemExit(
            f"state count mismatch: {len(states)} != {len(pdfs)}"
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output_path.with_suffix(".tmp")
    with temporary_output.open("w") as handle:
        for position, state in enumerate(states):
            row = dict(state)
            row["_distributed_generation"] = {
                "task": args.task,
                "worker": worker,
                "segment": segment,
                "position": position,
            }
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_output, args.output_path)
    print(
        f"wrote {len(states)} states for worker={worker} "
        f"segment={segment} task={args.task}"
    )


if __name__ == "__main__":
    main()
