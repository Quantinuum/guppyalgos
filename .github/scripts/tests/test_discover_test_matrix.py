"""Regression tests for complete, read-only CI test discovery."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "discover_test_matrix.py"
spec = importlib.util.spec_from_file_location("discover_test_matrix", SCRIPT)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_every_test_has_one_shard(tmp_path: Path) -> None:
    """Cover loose, nested, newly added, and alternative-pattern test files."""
    root = tmp_path / "tests"
    files = [
        "test_root.py",
        "algorithms/test_loose.py",
        "algorithms/new/test_nested.py",
        "primitives/new/deep/example_test.py",
        "new_category/test_new.py",
        "new_category/deep/test_deep.py",
        "notebooks/test_notebook.py",
        "empty/helper.py",
        "test_benchmark.json",
    ]
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    shards = [Path(p) for p in module.discover_test_paths(root)]
    assert all(p.is_file() for p in shards)
    for test in [root / name for name in files[:6]]:
        owners = [s for s in shards if s == test or s in test.parents]
        assert len(owners) == 1, str(test)
    assert not any("notebooks" in p.parts for p in shards)
    assert shards == [Path(p) for p in module.discover_test_paths(root)]
    after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_empty_suite_fails(tmp_path: Path) -> None:
    """Fail visibly rather than producing an empty passing matrix."""
    with pytest.raises(ValueError, match="No non-notebook tests found"):
        module.discover_test_paths(tmp_path)


def test_repository_coverage() -> None:
    """Require all current library tests to have exactly one matrix owner."""
    root = SCRIPT.parents[2] / "tests"
    shards = [Path(p) for p in module.discover_test_paths(root)]
    tests = [p for p in root.rglob("*.py") if module._is_test(p)]
    assert tests
    for test in tests:
        owners = [s for s in shards if s == test or s in test.parents]
        assert len(owners) == (0 if "notebooks" in test.relative_to(root).parts else 1)


def test_balancing_uses_individual_modules_and_keeps_new_files(tmp_path: Path) -> None:
    """Distribute heavy files without losing new, unmeasured tests."""
    root = tmp_path / "tests"
    heavy = root / "heavy"
    heavy.mkdir(parents=True)
    first = heavy / "test_one.py"
    second = heavy / "test_two.py"
    new = root / "test_new.py"
    for path in [first, second, new]:
        path.touch()
    paths = module.discover_test_paths(root)
    timings = {first.as_posix(): 120.0, second.as_posix(): 100.0}
    batches = module.balance_test_paths(paths, timings, 2)
    assigned = [p for batch in batches for p in batch["paths"]]
    assert sorted(assigned) == sorted(p.as_posix() for p in [first, second, new])
    assert [b["estimated_seconds"] for b in batches] == [120.0, 160.0]
    assert batches == module.balance_test_paths(list(reversed(paths)), timings, 2)


def test_six_balanced_batches(tmp_path: Path) -> None:
    """Pair long and short groups using the longest-first policy."""
    paths = []
    timings = {}
    for i, seconds in enumerate([60, 50, 40, 30, 20, 10, 50, 40, 30, 20, 10]):
        path = tmp_path / f"test_{i}.py"
        path.touch()
        paths.append(path.as_posix())
        timings[path.as_posix()] = seconds
    batches = module.balance_test_paths(paths, timings)
    assert len(batches) == 6
    assert all(b["estimated_seconds"] == 60 for b in batches)


def test_invalid_timings_fail() -> None:
    """Reject invalid weights rather than emitting a malformed matrix."""
    with pytest.raises(ValueError, match="finite"):
        module.balance_test_paths([], {"test.py": float("nan")})
    with pytest.raises(ValueError, match="Duplicate"):
        module.balance_test_paths(["tests/test_a.py", "tests/test_a.py"], {})


def test_repository_batches_cover_every_file_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the real benchmark cannot drop or duplicate library tests."""
    monkeypatch.chdir(SCRIPT.parents[2])
    timings = json.loads(Path("tests/test_benchmarks.json").read_text())["files"]
    batches = module.balance_test_paths(module.discover_test_paths(), timings)
    assert len(batches) == 6
    paths = [Path(p) for b in batches for p in b["paths"]]
    for test in Path("tests").rglob("*.py"):
        if module._is_test(test):
            owners = sum(p == test or p in test.parents for p in paths)
            assert owners == (0 if "notebooks" in test.parts else 1), str(test)
