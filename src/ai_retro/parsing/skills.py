"""Скилл — то, у чего есть SKILL.md; встроенные команды Claude Code скиллами не являются."""

import re
from pathlib import Path

_NAME = re.compile(r"[\w][\w.:-]*")


def find_skill(name: str, project: str, *, skills_dir: Path) -> Path | None:
    """Путь к SKILL.md: сначала `<проект>/.claude/skills`, затем глобальный корень."""
    if not _NAME.fullmatch(name):
        return None
    roots = (Path(project) / ".claude" / "skills", skills_dir)
    return next((p for root in roots if (p := root / name / "SKILL.md").is_file()), None)


def skill_exists(name: str, project: str, *, skills_dir: Path) -> bool:
    """SKILL.md есть в глобальном корне скиллов или в `<проект>/.claude/skills`."""
    return find_skill(name, project, skills_dir=skills_dir) is not None
