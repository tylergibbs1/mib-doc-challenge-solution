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
    parser.add_argument("--local-workers", type=int, default=4)
    args = parser.parse_args()

    worker, segment = divmod(args.task, args.segments_per_worker)
    pdfs = sorted(
        (str(path) for path in args.input_dir.glob("*.pdf")),
        key=lambda path: (os.path.getsize(path), Path(path).name),
    )
    if not pdfs:
        raise SystemExit("state segment has no PDFs")

    temporary = Path(tempfile.mkdtemp(prefix="mib-segment-", dir="/tmp"))
    worker_count = min(args.local_workers, len(pdfs), os.cpu_count() or 1)
    shards = [
        predict.Shard(temporary, index, pdfs[index::worker_count])
        for index in range(worker_count)
    ]
    while not all(shard.finished for shard in shards):
        time.sleep(predict.POLL_SECS)
        for shard in shards:
            shard.tick()
    extracted = predict._collect_states(shards, pdfs, complete=True)
    # _collect_states preserves the production runner's shard-major emission
    # order. CI parallelism is an implementation detail, so restore the
    # segment's original production-stream order before attaching positions.
    by_stem = {predict._state_stem(state): state for state in extracted}
    states = [by_stem[Path(pdf).stem] for pdf in pdfs]
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
