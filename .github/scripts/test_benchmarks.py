#!/usr/bin/env python3
"""Update the timing estimates used to balance CI test batches."""

import argparse
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

BENCHMARKS = Path("tests/test_benchmarks.json")


def read_timings(reports: list[Path], n_pytest_workers: int = 4) -> dict[str, float]:
    """Convert JUnit XML reports into estimated seconds per test module.

    Sum non-skipped test durations per file within each report, divide by
    n_pytest_workers (default four), then take the median across reports
    (such as Python versions). The worker count must be positive.
    This estimates workload for balancing batches, not exact elapsed time.
    Resolve files from JUnit file attributes or classnames relative to the repo
    root. Reject failed reports, unknown files, invalid durations, or no timings.
    Return a path-to-seconds mapping rounded to three decimal places.

    For example, if tests/test_add.py totals 40, 48, and 80 seconds in three
    reports, its estimates are 10, 12, and 20 seconds after dividing by four.
    The returned entry is {"tests/test_add.py": 12.0}, the median estimate.
    Passing n_pytest_workers=2 instead gives {"tests/test_add.py": 24.0}.
    """
    if n_pytest_workers < 1:
        raise ValueError("n_pytest_workers must be positive")
    samples: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        root = ET.parse(report).getroot()
        if root.find(".//failure") is not None or root.find(".//error") is not None:
            raise ValueError(f"Cannot benchmark failed tests: {report}")
        totals: dict[str, float] = defaultdict(float)
        for case in root.iter("testcase"):
            if case.find("skipped") is not None:
                continue
            path = case.get("file")
            if not path:
                parts = case.attrib["classname"].split(".")
                # JUnit classnames can end with a unittest/pytest class name.
                while parts and not Path("/".join(parts) + ".py").is_file():
                    parts.pop()
                path = "/".join(parts) + ".py"
            if not path.startswith("tests/") or not Path(path).is_file():
                raise ValueError(f"Unknown test file in {report}: {path}")
            seconds = float(case.attrib["time"])
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"Invalid test duration in {report}: {seconds}")
            totals[path] += seconds
        for path, seconds in totals.items():
            samples[path].append(seconds / n_pytest_workers)
    if not samples:
        raise ValueError("No test durations found")
    return {
        path: round(statistics.median(values), 3) for path, values in samples.items()
    }


def main(argv: list[str] | None = None) -> None:
    """Update tests/test_benchmarks.json from a report directory.

    Parse the directory from argv, or the command line when argv is None, and
    recursively read its XML reports. Replace measured file estimates while
    retaining unobserved entries, then write sorted JSON with an updated source
    description. Run from the repository root so test paths can be resolved.

    For example, `python3 .github/scripts/test_benchmarks.py test-timings` reads
    reports such as test-timings/python-3.12/test-results.xml. A measured entry
    for tests/test_add.py replaces its old estimate; other files keep theirs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    args = parser.parse_args(argv)
    data = json.loads(BENCHMARKS.read_text())
    data["files"].update(read_timings(sorted(args.reports.rglob("*.xml"))))
    data["files"] = dict(sorted(data["files"].items()))
    data["source"] = (
        "CI JUnit per-file test time divided by four workers; "
        "unobserved entries retained"
    )
    BENCHMARKS.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
