# Running tests

Run the tests affected by your change with `uv run pytest tests/path -q`.
Run the full library suite with `uv run pytest tests --ignore=tests/notebooks -n auto`,
and execute notebooks separately with `uv run pytest tests/notebooks -v`.
Notebook execution must remain sequential.

## Unitary modifier coverage

Each new capability must have explicit coverage. A separate test function for
each capability is not required: `assert_unitary_modifiers` checks forward
unitarity, dagger, control, and controlled dagger with one invocation:

```python
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, control, nat
from guppylang.std.quantum import qubit, rz
from guppyalgos.testing import assert_unitary_modifiers


@guppy.unitary
class rotation:
    @guppy
    def __call__(qs: array[qubit, 1]) -> None:
        rz(qs[0], angle(0.7))

    @guppy
    def daggered(qs: array[qubit, 1]) -> None:
        rz(qs[0], angle(-0.7))

    @guppy
    def controlled[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        with control(cs):
            rz(qs[0], angle(0.7))

    @guppy
    def ctrl_daggered[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        with control(cs):
            rz(qs[0], angle(-0.7))


def test_rotation_modifiers() -> None:
    assert_unitary_modifiers(rotation, 1)
```

Use this helper to test manually implemented custom modifiers, as above.
Automatically generated modifiers are the compiler's testing responsibility
and do not need additional modifier tests here.

The helper works with circuits that accept a single qubit array, including
wrappers around `@guppy.unitary` custom implementations. It checks one control qubit;
additional supported control counts need their own coverage. Failures name the
mode. Keep existing algorithm correctness tests, or pass `expected_unitary` to
compare the forward matrix with an independent reference (up to global phase).

Controlled references preserve the extracted forward matrix's phase relative to
the inactive branch. Do not phase-align that matrix before building a controlled
reference. The helper compares the full controlled matrix with one shared phase
alignment.
Each mode first checks matrix unitarity, then correctness, with distinct failure
messages for a nonunitary matrix and an incorrect unitary operation.

Use this helper only for small, fully unitary circuits because matrix extraction
scales exponentially. `endianness`, `n_extra_qubits`, and `threshold` configure
matrix ordering, extra simulator qubits, and absolute tolerance. Internal ancilla
must return to zero. Promised-input or measurement-cleanup routines still need
specialized tests.

## CI test discovery

`.github/workflows/ci.yml` runs library tests on Python 3.12, 3.13, and 3.14.
Notebooks run in a separate Python 3.13 job. Preview the test matrix with:

```sh
python3 .github/scripts/discover_test_matrix.py
```

Discovery recursively finds individual test modules (`test_*.py` and `*_test.py`)
at every directory depth. Files in parent directories are included too. The longest
estimated modules are assigned first to the least-loaded of six batches. Directories
are never passed alongside their children, so every test is included exactly once
per Python version. New files get a fallback estimate of 60 seconds.

`tests/test_benchmarks.json` records per-file workload estimates for four pytest
workers.
Initial estimates come from Python 3.14 CI log timestamps and include scheduling
overhead. Balance is approximate: one slow file can still dominate a batch, and
Python versions and runners differ. Discovery never runs tests or edits files.

The matrix has six batches per Python version (18 jobs), with at most six jobs
running concurrently. Each batch invokes pytest once with four workers (`-n 4`).
Notebooks remain sequential and docs build separately.

Each batch uploads JUnit timings. After all validation jobs pass, CI aggregates these
into a `test-benchmarks` artifact for inspection. Successful `main` runs also save
the estimates using GitHub Actions cache. Future runs restore the latest matching
cache, falling back to the checked-in JSON if none is available. PRs read the cache
but never save timings for future runs. The restore/save steps show cache activity
in the Actions logs; no GitHub CLI commands or extra tokens are needed.
The discovery job summary lists each batch's estimated workload.

To refresh the checked-in seed manually, download the `test-timings-*` artifacts
from a successful CI run into a directory, then run:

```sh
python3 .github/scripts/test_benchmarks.py /path/to/downloaded-artifacts
```

The updater uses median per-file totals across reports, divided by four workers,
and retains previous estimates for unobserved files. Review and commit the updated
JSON; CI uploads reports but never commits timing changes automatically.
