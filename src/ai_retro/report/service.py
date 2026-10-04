import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from ai_retro.classify.service import CATEGORIES, LOOP_MARKER
from ai_retro.db.models import ClassificationRow, EpisodeRow, PatternRow, ProposalRow
from ai_retro.errors import InvalidReportPathError, ModelCallError, NoClassificationsError
from ai_retro.llm import ModelCall
from ai_retro.report.schemas import GistItem, Pattern, ReportData

PATTERN_CATEGORIES = ("поправка", "инструкция")
MIN_EPISODES = 3
MIN_CONVERSATIONS = 2
TOP_N = 10
EXAMPLES = 3
EXAMPLE_LIMIT = 200
NO_DOMAIN = "без области"
PASTE_MARKER = "[вставка]"
# Ниже этого сходства суть эпизода с названием паттерна — повод перепроверить глазами.
RECHECK_SIMILARITY = 0.4
PROTECTED_NAMES = frozenset({"claude.md", "skill.md"})

SYSTEM_PROMPT = f"""{LOOP_MARKER}
Тебе дан список сутей эпизодов: «номер: суть». Сгруппируй номера по смыслу: в одну группу —
эпизоды, где пользователь поправил или сообщил одно и то же. Каждому номеру — не больше одной
группы; эпизоды без пары не включай. Для каждой группы дай короткое название.
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["title", "ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}

_PASTE_RE = re.compile(
    r"<pasted_content>.*?</pasted_content>|```.*?```|\[(?:Pasted text|Image)[^\]]*\]",
    re.DOTALL,
)


def load_report_data(
    engine: Engine, *, since: datetime | None, classifier_version: str
) -> tuple[list[GistItem], dict[str, dict[str, int]]]:
    """Классифицированные эпизоды периода: (кандидаты в паттерны, распределение по областям)."""
    stmt = (
        select(EpisodeRow, ClassificationRow)
        .join(ClassificationRow, ClassificationRow.episode_uuid == EpisodeRow.uuid)
        .where(ClassificationRow.classifier_version == classifier_version)
        .order_by(EpisodeRow.timestamp, EpisodeRow.uuid)
    )
    items: list[GistItem] = []
    distribution: dict[str, dict[str, int]] = {}
    with Session(engine) as session:
        for episode, cls in session.execute(stmt):
            if since is not None and _utc(episode.timestamp) < since:
                continue
            domain = episode.domain or NO_DOMAIN
            by_cat = distribution.setdefault(domain, {})
            by_cat[cls.category] = by_cat.get(cls.category, 0) + 1
            if cls.category in PATTERN_CATEGORIES:
                items.append(
                    GistItem(
                        uuid=episode.uuid,
                        conversation_id=episode.conversation_id,
                        project=episode.project,
                        domain=domain,
                        category=cls.category,
                        gist=cls.gist,
                        text=episode.text,
                        timestamp=episode.timestamp,
                    )
                )
    if not distribution:
        raise NoClassificationsError
    return items, distribution


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def group_gists(call: ModelCall, items: Sequence[GistItem]) -> list[Pattern]:
    """Один вызов модели по списку сутей; порог паттерна применяется здесь, детерминированно."""
    if not items:
        return []
    prompt = "\n".join(f"{i}: {item.gist}" for i, item in enumerate(items))
    raw = call(prompt)
    try:
        groups = json.loads(raw)["groups"]
        parsed = [(str(g["title"]), [int(n) for n in g["ids"]]) for g in groups]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ModelCallError(f"Группировка вернула ответ не по схеме: {raw[:100]!r}") from exc

    used: set[int] = set()
    patterns: list[Pattern] = []
    for title, ids in parsed:
        members = [n for n in dict.fromkeys(ids) if 0 <= n < len(items) and n not in used]
        used.update(members)
        episodes = tuple(items[n] for n in members)
        pattern = Pattern(title=title.strip(), episodes=episodes)
        if len(episodes) >= MIN_EPISODES and len(pattern.conversations) >= MIN_CONVERSATIONS:
            patterns.append(pattern)
    return sorted(patterns, key=lambda p: (-len(p.episodes), p.title))


def save_patterns(engine: Engine, patterns: Sequence[Pattern], *, now: datetime) -> None:
    """Заменяет прежние паттерны, на которые ещё нет предложений."""
    with Session(engine) as session:
        proposed = select(ProposalRow.pattern_id)
        session.execute(delete(PatternRow).where(PatternRow.id.not_in(proposed)))
        for p in patterns:
            session.add(
                PatternRow(
                    gist=p.title,
                    episode_uuids=[e.uuid for e in p.episodes],
                    created_at=now,
                )
            )
        session.commit()


def build_report(
    engine: Engine,
    call: ModelCall,
    *,
    since: datetime | None,
    now: datetime,
    classifier_version: str,
) -> ReportData:
    items, distribution = load_report_data(
        engine, since=since, classifier_version=classifier_version
    )
    patterns = group_gists(call, items)
    save_patterns(engine, patterns, now=now)
    return ReportData(since=since, distribution=distribution, patterns=tuple(patterns))


def mask_pastes(text: str) -> str:
    """Вставки (<pasted_content>, блоки кода, плейсхолдеры) показываются маркером с длиной."""
    return _PASTE_RE.sub(lambda m: f"[вставка, {len(m.group(0))} симв.]", text)


def example_text(text: str) -> str:
    flat = " ".join(mask_pastes(text).split())
    return flat if len(flat) <= EXAMPLE_LIMIT else flat[: EXAMPLE_LIMIT - 1] + "…"


def recheck_count(pattern: Pattern) -> int:
    """Эпизоды, чья суть мало похожа на название паттерна: кандидаты в «фальшивые».

    Это прокси по сходству формулировок, не оценка модели (см. ADR 0004).
    """
    title = pattern.title.lower()
    return sum(
        1
        for e in pattern.episodes
        if SequenceMatcher(None, title, e.gist.lower()).ratio() < RECHECK_SIMILARITY
    )


def _examples(pattern: Pattern) -> list[GistItem]:
    """Примеры из разных бесед в первую очередь."""
    seen: set[str] = set()
    first: list[GistItem] = []
    rest: list[GistItem] = []
    for e in pattern.episodes:
        (rest if e.conversation_id in seen else first).append(e)
        seen.add(e.conversation_id)
    return (first + rest)[:EXAMPLES]


def render_report(data: ReportData) -> str:
    period = f"с {data.since:%Y-%m-%d}" if data.since else "за всё время"
    lines = [f"# Отчёт ai-retro ({period})", "", "## Категории по областям", ""]
    lines += ["| Область | " + " | ".join(CATEGORIES) + " |"]
    lines += ["|---|" + "---|" * len(CATEGORIES)]
    for domain in sorted(data.distribution):
        counts = data.distribution[domain]
        lines.append(
            f"| {domain} | " + " | ".join(str(counts.get(c, 0)) for c in CATEGORIES) + " |"
        )
    lines += ["", "## Паттерны", ""]
    if not data.patterns:
        lines += [
            (
                f"Паттернов нет: нужна группа из {MIN_EPISODES}+ эпизодов "
                f"из {MIN_CONVERSATIONS}+ бесед."
            ),
            "",
        ]
    for p in data.patterns:
        lines += [
            f"### {p.title}",
            "",
            f"- эпизодов: {len(p.episodes)}, бесед: {len(p.conversations)}",
            f"- области: {', '.join(p.domains)}",
            f"- проекты: {', '.join(p.projects)}",
            "- примеры:",
        ]
        lines += [
            f"  - «{example_text(e.text)}» — беседа `{e.conversation_id}`" for e in _examples(p)
        ]
        lines.append("")
    lines += [f"## Топ-{TOP_N}: что перепроверить глазами", ""]
    for i, p in enumerate(data.patterns[:TOP_N], 1):
        lines.append(
            f"{i}. {p.title} — эпизодов: {len(p.episodes)}, "
            f"перепроверить: {recheck_count(p)} из {len(p.episodes)}"
        )
    lines += [
        "",
        (
            "Число «перепроверить» — грубая оценка по сходству формулировок сути и названия, "
            "а не вывод модели: смотрите эпизоды глазами."
        ),
    ]
    return "\n".join(lines) + "\n"


def check_report_path(path: Path) -> None:
    """Отчёт не пишется поверх CLAUDE.md и SKILL.md (без учёта регистра)."""
    if path.name.casefold() in PROTECTED_NAMES:
        raise InvalidReportPathError(path)


def write_report(path: Path, text: str) -> None:
    check_report_path(path)
    path.write_text(text, encoding="utf-8")
