"""Tests for repo-local agent skill wiring."""

from pathlib import Path


def _frontmatter_block(markdown: str) -> str:
    lines = markdown.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return ""

    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i])

    return ""


def test_agents_md_references_repo_skill() -> None:
    """Assert AGENTS.md points to the repo-local skill path."""
    repo_root = Path(__file__).resolve().parents[2]
    agents_md = repo_root / "AGENTS.md"
    skill_md = repo_root / "agents" / "skills" / "guppyalgos" / "SKILL.md"

    assert agents_md.exists(), "Expected AGENTS.md at repository root."
    assert skill_md.exists(), (
        "Expected skill file at agents/skills/guppyalgos/SKILL.md."
    )

    agents_text = agents_md.read_text(encoding="utf-8")
    assert "## Skills" in agents_text
    assert "./agents/skills/guppyalgos/SKILL.md" in agents_text


def test_skill_md_has_required_frontmatter_fields() -> None:
    """Assert skill frontmatter includes required trigger metadata."""
    repo_root = Path(__file__).resolve().parents[2]
    skill_md = repo_root / "agents" / "skills" / "guppyalgos" / "SKILL.md"
    skill_text = skill_md.read_text(encoding="utf-8")
    frontmatter = _frontmatter_block(skill_text)

    assert frontmatter, "SKILL.md must start with YAML frontmatter."
    assert "name:" in frontmatter, "SKILL.md frontmatter must define name."
    assert "description:" in frontmatter, (
        "SKILL.md frontmatter must define description."
    )
