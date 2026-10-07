"""Предложения правок: цель выбирают правила, текст пишет модель (ADR 0007)."""

import json
from collections import Counter
from pathlib import Path

from ai_retro.classify.service import LOOP_MARKER
from ai_retro.domain.service import is_scratch
from ai_retro.errors import ModelCallError
from ai_retro.llm import ModelCall
from ai_retro.report.schemas import (
    PROPOSED,
    TARGET_GLOBAL,
    TARGET_PROJECT,
    TARGET_SKILL,
    Pattern,
    Proposal,
    Target,
)
from ai_retro.report.text import example_text

GLOBAL_CLAUDE_MD = "~/.claude/CLAUDE.md"
PROMPT_EXAMPLES = 5
PROMPT_TEXT_LIMIT = 300

STATUS_LABEL = {PROPOSED: "предложено"}
KIND_EDIT = "edit"
KIND_NEW_SKILL = "new_skill"

SYSTEM_PROMPT = f"""{LOOP_MARKER}
Ты пишешь правку для файла инструкций агента по повторяющейся поправке или инструкции
пользователя. Тебе дают название паттерна, суть и реплики эпизодов, целевой файл.
Верни:
- kind: «edit» — правка файла; «new_skill» — повторяющееся действие лучше оформить новым
  скиллом, чем правилом (тогда файла нет, section оставь пустым, а в text опиши скилл);
- section: заголовок раздела целевого файла, куда добавить текст;
- text: готовый текст правки, как он будет лежать в файле: коротко, повелительно, без ссылок на
  конкретные беседы;
- rationale: одно-два предложения, почему правка нужна, со ссылкой на повтор.
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": [KIND_EDIT, KIND_NEW_SKILL]},
        "section": {"type": "string"},
        "text": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["kind", "section", "text", "rationale"],
    "additionalProperties": False,
}


def skill_file(name: str) -> str:
    return f"~/.claude/skills/{name}/SKILL.md"


def choose_target(pattern: Pattern) -> Target:
    """Куда нацелить паттерн; чистая функция от эпизодов.

    Скилл — если с одним скиллом связано больше половины эпизодов; иначе беседы одного проекта
    — CLAUDE.md проекта; разные проекты и любой scratch — глобальный CLAUDE.md.
    """
    episodes = pattern.episodes
    skills = Counter(e.skill for e in episodes if e.skill)
    if skills:
        (skill, count), *_ = sorted(skills.items(), key=lambda kv: (-kv[1], kv[0]))
        if count * 2 > len(episodes):
            return Target(TARGET_SKILL, skill_file(skill))
    projects = {e.project for e in episodes}
    if len(projects) == 1 and not any(is_scratch(p) for p in projects):
        (project,) = projects
        return Target(TARGET_PROJECT, str(Path(project) / "CLAUDE.md"))
    return Target(TARGET_GLOBAL, GLOBAL_CLAUDE_MD)


def proposal_prompt(pattern: Pattern, target: Target) -> str:
    lines = [
        f"Паттерн: {pattern.title}",
        f"Целевой файл: {target.file}",
        f"Эпизодов: {len(pattern.episodes)}, бесед: {len(pattern.conversations)}",
        "Эпизоды:",
    ]
    for e in pattern.episodes[:PROMPT_EXAMPLES]:
        lines.append(
            f"- суть: {e.gist}; реплика: «{example_text(e.text, limit=PROMPT_TEXT_LIMIT)}»"
        )
    return "\n".join(lines)


def generate_proposal(call: ModelCall, pattern: Pattern) -> Proposal:
    target = choose_target(pattern)
    raw = call(proposal_prompt(pattern, target))
    try:
        data = json.loads(raw)
        kind, section, text, rationale = (
            str(data["kind"]),
            str(data["section"]).strip(),
            str(data["text"]).strip(),
            str(data["rationale"]).strip(),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ModelCallError(f"Предложение вернулось не по схеме: {raw[:100]!r}") from exc
    if kind not in (KIND_EDIT, KIND_NEW_SKILL) or not text:
        raise ModelCallError(f"Предложение вернулось не по схеме: {raw[:100]!r}")
    if kind == KIND_NEW_SKILL:
        return Proposal(target_file="", section="", text=text, rationale=rationale, status=PROPOSED)
    return Proposal(
        target_file=target.file, section=section, text=text, rationale=rationale, status=PROPOSED
    )


def render_proposal(proposal: Proposal) -> list[str]:
    where = (
        f"`{proposal.target_file}`, раздел «{proposal.section}»"
        if proposal.target_file
        else "новый скилл (целевого файла нет, предложение только текстом)"
    )
    lines = [
        f"- предложение ({STATUS_LABEL.get(proposal.status, proposal.status)}): {where}",
        "  - текст:",
    ]
    lines += [f"    > {line}" if line else "    >" for line in proposal.text.splitlines()]
    lines.append(f"  - обоснование: {proposal.rationale}")
    return lines
