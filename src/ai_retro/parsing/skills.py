"""Скилл — то, у чего есть SKILL.md; встроенные команды Claude Code скиллами не являются."""

import re
from pathlib import Path

_NAME = re.compile(r"[\w][\w.:-]*")


def skill_exists(name: str, project: str, *, skills_dir: Path) -> bool:
    """SKILL.md есть в глобальном корне скиллов или в `<проект>/.claude/skills`."""
    if not _NAME.fullmatch(name):
        return False
    roots = (skills_dir, Path(project) / ".claude" / "skills")
    return any((root / name / "SKILL.md").is_file() for root in roots)
