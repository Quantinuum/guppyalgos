"""Check that public sync preserves the internal branch's private directories."""

import subprocess
import textwrap
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows/sync-internal.yml"
PRIVATE_PATHS = ("guppyalgos/internal", "tests/internal", "examples/internal")


def git(repo: Path, *args: str) -> str:
    """Run Git in an isolated fixture repository."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def write(repo: Path, name: str, content: str) -> None:
    """Create a fixture file and its parent directories."""
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def sync_commands() -> tuple[str, str]:
    """Execute the workflow's actual source filtering and destination replacement."""
    script = textwrap.dedent(WORKFLOW.read_text().split("        run: |\n", 1)[1])
    paths = script.split("preserved_paths=", 1)[1].splitlines()[0]
    prepare = "preserved_paths=" + paths + "\n"
    prepare += "git checkout -B sync-source origin/main\n"
    prepare += next(
        line
        for line in script.splitlines()
        if "git rm" in line and "preserved_paths" in line
    )
    replace = script.split("if git fetch internal main-internal; then\n", 1)[1]
    replace = textwrap.dedent(replace.split("git add -A", 1)[0])
    return prepare, replace


@pytest.mark.parametrize(
    "private_paths", [(), *[(p,) for p in PRIVATE_PATHS], PRIVATE_PATHS]
)
def test_sync_preserves_private_trees(
    tmp_path: Path, private_paths: tuple[str, ...]
) -> None:
    """Keep private tree contents exact while updating and deleting public files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "public")
    git(repo, "config", "user.name", "Sync Test")
    git(repo, "config", "user.email", "sync-test@example.com")
    git(repo, "config", "commit.gpgsign", "false")
    write(repo, "shared.txt", "old public content\n")
    write(repo, "stale.txt", "removed upstream\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Base")
    git(repo, "checkout", "-b", "internal-source")
    for path in private_paths:
        write(repo, f"{path}/nested/private.txt", "private content\n")
        write(repo, f"{path}/conflict.txt", "private version\n")
    git(repo, "add", ".")
    git(repo, "commit", "--allow-empty", "-m", "Private files")
    trees = {path: git(repo, "rev-parse", f"HEAD:{path}") for path in private_paths}
    git(repo, "checkout", "public")
    write(repo, "shared.txt", "new public content\n")
    write(repo, "new.txt", "added upstream\n")
    git(repo, "rm", "stale.txt")
    for path in PRIVATE_PATHS:
        write(repo, f"{path}/conflict.txt", "public version\n")
        write(repo, f"{path}/public-only.txt", "must not enter private directory\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Public updates")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    prepare, replace = sync_commands()
    previous_tree = None
    for _ in range(2):
        subprocess.run(
            [
                "bash",
                "-c",
                "set -euo pipefail\n"
                + prepare
                + '\ngit commit -m "Prepare" || true\n'
                + "git fetch . internal-source\n"
                + replace
                + '\ngit add -A\ngit commit -m "Sync" || true\n',
            ],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        assert (repo / "shared.txt").read_text() == "new public content\n"
        assert (repo / "new.txt").exists()
        assert not (repo / "stale.txt").exists()
        for path in PRIVATE_PATHS:
            if path in trees:
                assert git(repo, "rev-parse", f"HEAD:{path}") == trees[path]
            else:
                assert not (repo / path).exists()
        current_tree = git(repo, "rev-parse", "HEAD^{tree}")
        if previous_tree is not None:
            assert current_tree == previous_tree
        previous_tree = current_tree
        git(repo, "branch", "-f", "internal-source", "HEAD")
