# Running tests

Run the tests affected by your change with `uv run pytest tests/path -q`.
Run the full library suite with `uv run pytest tests --ignore=tests/notebooks -n auto`,
and execute notebooks separately with `uv run pytest tests/notebooks -v`.
Notebook execution must remain sequential.

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
