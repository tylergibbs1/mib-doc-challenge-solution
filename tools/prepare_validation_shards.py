#!/usr/bin/env python3
"""Partition public validation PDFs for distributed state extraction.

The production runner sorts by (size, filename), assigns every fourth PDF to
one of four workers, and later emits worker 0's stream followed by workers
1-3.  This tool preserves that exact ordering while splitting each worker
stream into four contiguous CI tasks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def contiguous_bounds(length: int, parts: int, index: int) -> tuple[int, int]:
    base, remainder = divmod(length, parts)
    start = index * base + min(index, remainder)
    stop = start + base + (1 if index < remainder else 0)
    return start, stop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--production-workers", type=int, default=4)
    parser.add_argument("--segments-per-worker", type=int, default=4)
    parser.add_argument("--expected-count", type=int, default=5000)
    args = parser.parse_args()

    pdfs = sorted(
        args.input_dir.glob("*.pdf"),
        key=lambda path: (path.stat().st_size, path.name),
    )
    if len(pdfs) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} PDFs, found {len(pdfs)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)

    manifest = []
    for worker in range(args.production_workers):
        stream = pdfs[worker :: args.production_workers]
        for segment in range(args.segments_per_worker):
            task = worker * args.segments_per_worker + segment
            start, stop = contiguous_bounds(
                len(stream), args.segments_per_worker, segment
            )
            task_dir = args.output_dir / str(task)
            task_dir.mkdir()
            selected = stream[start:stop]
            for source in selected:
                os.link(source, task_dir / source.name)
            manifest.append(
                {
                    "task": task,
                    "worker": worker,
                    "segment": segment,
                    "stream_start": start,
                    "stream_stop": stop,
                    "count": len(selected),
                    "first": selected[0].name if selected else None,
                    "last": selected[-1].name if selected else None,
                }
            )

    if sum(row["count"] for row in manifest) != len(pdfs):
        raise SystemExit("partition lost or duplicated PDFs")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
