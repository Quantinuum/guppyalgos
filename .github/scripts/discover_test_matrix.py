#!/usr/bin/env python3
"""Discover deterministic test shards without running tests or modifying the repo."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TypedDict


class TestBatch(TypedDict):
    """Paths and estimated workload assigned to a runner."""

    batch: int
    paths: list[str]
    estimated_seconds: float


def _is_test(path: Path) -> bool:
    """Return whether the path is a file named test_*.py or *_test.py.

    For example, an existing tests/test_add.py matches; tests/helpers.py does not.
    """
    return path.is_file() and (path.match("test_*.py") or path.match("*_test.py"))


def discover_test_paths(tests_dir: Path = Path("tests")) -> list[str]:
    """Return sorted test module paths found recursively under tests_dir.

    Include files at every directory depth, excluding notebook directories
    because notebooks run in a separate CI job. Paths use forward slashes and
    retain the supplied root. Raise ValueError if no library test files exist.

    For example, tests/test_a.py and tests/arithmetic/test_b.py are both returned,
    but tests/notebooks/test_examples.py is excluded. Each returned path is a
    single file, so passing the paths to pytest does not collect a directory twice.
    """
    files = sorted(
        p
        for p in tests_dir.rglob("*.py")
        if _is_test(p) and "notebooks" not in p.relative_to(tests_dir).parts
    )
    if not files:
        raise ValueError("No non-notebook tests found")
    return [path.as_posix() for path in files]


def balance_test_paths(
    paths: list[str], timings: dict[str, float], batch_count: int = 6
) -> list[TestBatch]:
    """Distribute test modules across batches using estimated durations.

    Assign each file once, longest first, to the batch with the lowest total.
    Unknown files receive a 60-second estimate; ties use paths and batch numbers
    for reproducible results. Return up to batch_count batches with their paths
    and total estimated seconds, or an empty list when paths is empty.
    Reject duplicate paths, invalid durations, and nonpositive batch counts.

    For example, files a.py, b.py, and c.py estimated at 90, 60, and 30 seconds
    form two equally weighted batches: [a.py] and [b.py, c.py]. The first result
    is {"batch": 1, "paths": ["a.py"], "estimated_seconds": 90.0}.
    """
    if batch_count < 1:
        raise ValueError("batch_count must be positive")
    if any(not math.isfinite(t) or t < 0 for t in timings.values()):
        raise ValueError("Benchmark durations must be finite and nonnegative")

    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate test paths")

    def estimate(path: str) -> float:
        """Return the file's estimated seconds, defaulting to 60 if unknown.

        For example, a newly added test module absent from timings gets 60.0.
        """
        return timings.get(path, 60.0)

    batches: list[TestBatch] = [
        {"batch": i + 1, "paths": [], "estimated_seconds": 0.0}
        for i in range(min(batch_count, len(paths)))
    ]
    for path in sorted(paths, key=lambda p: (-estimate(p), p)):
        batch = min(batches, key=lambda b: (b["estimated_seconds"], b["batch"]))
        batch["paths"].append(path)
        batch["estimated_seconds"] += estimate(path)
    for batch in batches:
        batch["estimated_seconds"] = round(batch["estimated_seconds"], 2)
    return batches


def main() -> None:
    """Build six test batches from tests/test_benchmarks.json.

    Always print the matrix as batches=<JSON> for local inspection. In CI, also
    append it to GITHUB_OUTPUT and write a workload table to GITHUB_STEP_SUMMARY
    when those environment variables are set. No tests are executed.

    For example, run `python3 .github/scripts/discover_test_matrix.py` locally
    to inspect the batches. CI uses the JSON list as a matrix: each batch and
    Python version gets its own runner, which receives that batch's paths.
    """
    benchmark = json.loads(Path("tests/test_benchmarks.json").read_text())
    batches = balance_test_paths(discover_test_paths(), benchmark["files"])
    value = json.dumps(batches)
    line = f"batches={value}\n"
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as output_file:
            output_file.write(line)
    print(line, end="")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as summary_file:
            summary_file.write(
                "| Batch | Test modules | Estimated minutes |\n|---|---:|---:|\n"
            )
            for batch in batches:
                summary_file.write(
                    f"| {batch['batch']} | {len(batch['paths'])} | "
                    f"{batch['estimated_seconds'] / 60:.1f} |\n"
                )


if __name__ == "__main__":
    main()
