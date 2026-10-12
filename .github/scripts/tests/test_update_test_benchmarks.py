"""Checks for benchmark aggregation from CI artifacts."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "test_benchmarks.py"
spec = importlib.util.spec_from_file_location("update_test_benchmarks", SCRIPT)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_median_totals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Aggregate all cases per file before taking the cross-report median."""
    monkeypatch.chdir(tmp_path)
    Path("tests").mkdir()
    Path("tests/test_example.py").touch()
    reports = []
    for i, seconds in enumerate([4, 12, 20]):
        p = tmp_path / f"{i}.xml"
        p.write_text(
            "<testsuites><testsuite>"
            f'<testcase classname="tests.test_example" time="{seconds}"/>'
            '<testcase classname="tests.test_example" time="4"/>'
            "</testsuite></testsuites>"
        )
        reports.append(p)
    assert module.read_timings(reports) == {"tests/test_example.py": 4.0}
    assert module.read_timings(reports, n_pytest_workers=2) == {
        "tests/test_example.py": 8.0
    }


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_count(workers: int) -> None:
    """Reject worker counts that cannot scale timing estimates."""
    with pytest.raises(ValueError, match="n_pytest_workers must be positive"):
        module.read_timings([], n_pytest_workers=workers)


def test_failed_report_rejected(tmp_path: Path) -> None:
    """A failed run must not silently overwrite performance estimates."""
    p = tmp_path / "failed.xml"
    p.write_text("<testsuite><testcase><failure/></testcase></testsuite>")
    with pytest.raises(ValueError, match="failed"):
        module.read_timings([p])


def test_classname_and_invalid_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve class-based tests to their module and reject invalid measurements."""
    monkeypatch.chdir(tmp_path)
    Path("tests").mkdir()
    Path("tests/test_example.py").touch()
    p = Path("report.xml")
    p.write_text(
        '<testsuite><testcase classname="tests.test_example.TestClass" '
        'time="8"/></testsuite>'
    )
    assert module.read_timings([p]) == {"tests/test_example.py": 2.0}
    p.write_text(
        '<testsuite><testcase classname="tests.test_example" time="nan"/></testsuite>'
    )
    with pytest.raises(ValueError, match="Invalid test duration"):
        module.read_timings([p])


def test_update_command_preserves_unobserved_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared CLI updates measured modules without dropping other estimates."""
    monkeypatch.chdir(tmp_path)
    Path("tests").mkdir()
    Path("tests/test_example.py").touch()
    Path("reports").mkdir()
    Path("reports/run.xml").write_text(
        '<testsuite><testcase classname="tests.test_example" time="8"/></testsuite>'
    )
    Path("tests/test_benchmarks.json").write_text(
        json.dumps({"files": {"tests/test_other.py": 30}})
    )
    module.main(["reports"])
    assert json.loads(Path("tests/test_benchmarks.json").read_text())["files"] == {
        "tests/test_example.py": 2.0,
        "tests/test_other.py": 30,
    }
