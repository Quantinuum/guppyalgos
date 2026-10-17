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
import numpy as np
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
    expected = np.diag(np.exp(np.pi * np.array([-0.35j, 0.35j])))
    assert_unitary_modifiers(rotation, 1, expected_unitary=expected)
```

Use this helper to test manually implemented custom modifiers, as above.
Automatically generated modifiers are the compiler's testing responsibility
and do not need additional modifier tests here.

The helper works with circuits that accept a single qubit array, including
wrappers around `@guppy.unitary` custom implementations. It checks one control qubit;
additional supported control counts need their own coverage. Failures name the
mode. The required `expected_unitary` checks the full forward matrix up to one
overall global phase and uses the reference's exact phase for full controlled-matrix
expectations. The dagger is compared with the reference adjoint. The reference
must be independent of the controlled result being tested.

Control checks extract the complete matrix over the control and target registers.
The expected matrix is built with `np.kron`: identity when the control is off and
the supplied reference (or its adjoint) when it is on. This covers both control
inputs, control-register preservation, and relative phase between branches.
Only one shared overall global phase is ignored for the complete matrix.
A reference is always required so arbitrary compilation phases from the
extracted forward matrix cannot determine the controlled expectation.

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
