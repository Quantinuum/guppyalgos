# Contributing

We welcome contributions to guppyalgos! To report a bug, request a feature, or
propose an algorithm, open [an issue](https://github.com/Quantinuum/guppyalgos/issues/new).
To contribute code, [fork](https://github.com/Quantinuum/guppyalgos/fork) the
repository and open a pull request against the `main` branch. Pull requests must
pass all CI checks before they are merged, including the
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) check for
the pull request title.

Open an issue before starting substantial work. Include a minimal reproducer for
bugs and a paper reference for new algorithms.

## Development setup

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). From a source
checkout, install the development dependencies and Git hooks:

```sh
uv sync --extra dev-dependencies
uv run prek install
```

## Pull requests and gh stack

Keep each PR focused on one logical change. We use
[`gh stack`](https://github.com/github/gh-stack) for dependent changes: each branch
builds on the previous one, giving reviewers a small diff for each layer. A single
PR is fine for an independent change.

Install and authenticate the [GitHub CLI](https://cli.github.com/), then install
the extension:

```sh
gh auth login
gh extension install github/gh-stack
```

For a change with two dependent parts:

```sh
gh stack init --base main feature/core
# Make and commit the first part.
gh stack add feature/integration
# Make and commit the dependent part.
gh stack submit
```

Use `gh stack sync` to update the stack after changes or merges. See the
[gh stack guide](https://github.github.com/gh-stack/introduction/overview/)
for rebasing, navigation, and other commands.

Open draft PRs while work is in progress. Link each PR to exactly one issue with
`Closes #123`, and use a Conventional Commit title such as `feat: add an algorithm`
or `fix: correct a rotation`. Describe the behavior change and how you tested it.

## Before requesting review

- Add or update tests for behavior changes, and update docs for public API changes.
- Use American English in prose and identifiers; follow nearby code conventions.
- Cite the source paper when implementing an algorithm.
- Run the affected tests first, then the repository checks:

```sh
uv run pytest tests/path/to/test_module.py -q
uv run prek run --all-files
```

Replace the example test path with the relevant files. For changes that affect the
whole library, follow the full-suite commands in [tests/README.md](tests/README.md).
The hooks check formatting, lint, types, spelling, and file hygiene.

## Continuous integration

PRs targeting any branch run the same validation, including layers of a `gh stack`.
CI checks code, tests Python 3.12–3.14, executes notebooks sequentially, builds the
docs, and checks an installed wheel. Validation uses public dependencies and the
read-only built-in GitHub token; contributors do not need CI secrets or a PAT.
Vendored skill material is excluded from formatting and spelling hooks.

The standalone `test-docs-build.yml` workflow provides the `Test sphinx docs` check.
For branch protection, require `CI passed`, `Test sphinx docs`, and
`Conventional Commit title`. `CI passed` covers the jobs in `ci.yml` and fails if
any of them fails, is canceled, or is skipped. All three checks support the merge
queue. Dependency review runs separately on every
PR and rejects newly introduced high or critical severity vulnerabilities; keep
it as a PR review gate, since it does not run on merge-queue events.

## Releases

Release Please manages versions and the changelog from Conventional Commit titles.
Before `1.0.0`, `fix:` and `feat:` increment the patch version; a breaking change
marked with `!` or a `BREAKING CHANGE:` footer increments the minor version.

 Reserve `Release-As: 1.0.0` for the deliberate decision to declare the public
 API stable.

### Publishing a release

The PyPI project is `guppyalgos`. Configure its GitHub Actions trusted publisher
with owner `Quantinuum`, repository `guppyalgos`, workflow `build_wheels.yml`,
and environment `pypi`. When renaming the GitHub repository, update the trusted
publisher in PyPI's project publishing settings before publishing another release.

1. Review and merge the Release Please PR. It creates the version tag and GitHub
   release.
2. `Build wheels` validates the universal wheel and source archive. It starts from
   the published release, or can be run manually for an existing version tag.
3. The `pypi` job attaches them to the GitHub release and publishes them to PyPI.

For a retry, manually rerun `Build wheels` for the same tag. Published PyPI files
cannot be replaced; release a new version to correct a package. Documentation builds
separately and is not deployed by this workflow.
