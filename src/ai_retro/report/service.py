import json
import os
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from ai_retro.classify.service import CATEGORIES, LOOP_MARKER
from ai_retro.db.models import ClassificationRow, EpisodeRow, PatternRow, ProposalRow
from ai_retro.errors import (
    InvalidReportPathError,
    ModelCallError,
    NoClassificationsError,
    ReportWriteError,
)
from ai_retro.llm import ModelCall
from ai_retro.parsing.skills import skill_exists
from ai_retro.report.proposals import generate_proposal, render_proposal
from ai_retro.report.schemas import GistItem, Pattern, Proposal, ReportData
from ai_retro.report.text import example_text

PATTERN_CATEGORIES = ("поправка", "инструкция")
MIN_EPISODES = 3
MIN_CONVERSATIONS = 2
TOP_N = 10
EXAMPLES = 3
NO_DOMAIN = "без области"
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


def load_report_data(
    engine: Engine, *, since: datetime | None, classifier_version: str, skills_dir: Path
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
        skill_calls = _skill_calls(session)
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
                        skill=_existing_skill(
                            _linked_skill(skill_calls, episode), episode.project, skills_dir
                        ),
                    )
                )
    if not distribution:
        raise NoClassificationsError
    return items, distribution


def _skill_calls(session: Session) -> dict[str, list[tuple[datetime, str]]]:
    """Вызовы скиллов по беседам, по времени; период отчёта не режет: вызов мог быть раньше."""
    stmt = (
        select(EpisodeRow.conversation_id, EpisodeRow.timestamp, EpisodeRow.skill)
        .where(EpisodeRow.skill.is_not(None))
        .order_by(EpisodeRow.timestamp, EpisodeRow.uuid)
    )
    calls: dict[str, list[tuple[datetime, str]]] = {}
    for conversation_id, timestamp, skill in session.execute(stmt):
        if skill:
            calls.setdefault(conversation_id, []).append((_utc(timestamp), skill))
    return calls


def _linked_skill(calls: dict[str, list[tuple[datetime, str]]], episode: EpisodeRow) -> str | None:
    """Скилл самого эпизода или последний вызванный в беседе до него."""
    if episode.skill:
        return episode.skill
    when = _utc(episode.timestamp)
    earlier = [skill for ts, skill in calls.get(episode.conversation_id, []) if ts < when]
    return earlier[-1] if earlier else None


def _existing_skill(name: str | None, project: str, skills_dir: Path) -> str | None:
    """Цель-скилл только с SKILL.md на диске; иначе правила цели идут к CLAUDE.md (ADR 0010)."""
    if name and skill_exists(name, project, skills_dir=skills_dir):
        return name
    return None


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


def kept_proposals(engine: Engine) -> dict[frozenset[str], Proposal]:
    """Предложения, уже лежащие в базе, по набору эпизодов паттерна."""
    stmt = select(PatternRow, ProposalRow).join(
        ProposalRow, ProposalRow.pattern_id == PatternRow.id
    )
    with Session(engine) as session:
        return {
            frozenset(pattern.episode_uuids): Proposal(
                target_file=row.target_file,
                section=row.section,
                text=row.text,
                rationale=row.rationale,
                status=row.status,
            )
            for pattern, row in session.execute(stmt)
        }


def save_patterns(engine: Engine, patterns: Sequence[Pattern], *, now: datetime) -> None:
    """Заменяет прежние паттерны без предложений; паттерн с предложением пишется со своим.

    Паттерн, чей набор эпизодов уже лежит в базе с предложением, не дублируется.
    """
    with Session(engine) as session:
        proposed = select(ProposalRow.pattern_id)
        session.execute(delete(PatternRow).where(PatternRow.id.not_in(proposed)))
        have = {
            frozenset(episode_uuids)
            for episode_uuids in session.scalars(select(PatternRow.episode_uuids))
        }
        for p in patterns:
            uuids = [e.uuid for e in p.episodes]
            if frozenset(uuids) in have:
                continue
            row = PatternRow(gist=p.title, episode_uuids=uuids, created_at=now)
            session.add(row)
            if p.proposal is not None:
                session.flush()
                session.add(
                    ProposalRow(
                        pattern_id=row.id,
                        target_file=p.proposal.target_file,
                        section=p.proposal.section,
                        text=p.proposal.text,
                        rationale=p.proposal.rationale,
                        status=p.proposal.status,
                    )
                )
        session.commit()


def with_proposals(
    engine: Engine, proposal_call: ModelCall, patterns: Sequence[Pattern]
) -> list[Pattern]:
    """Каждому паттерну — предложение: из базы, если оно уже есть, иначе новое от модели."""
    kept = kept_proposals(engine)
    result: list[Pattern] = []
    for p in patterns:
        proposal = kept.get(frozenset(e.uuid for e in p.episodes))
        result.append(replace(p, proposal=proposal or generate_proposal(proposal_call, p)))
    return result


def build_report(
    engine: Engine,
    call: ModelCall,
    proposal_call: ModelCall,
    *,
    since: datetime | None,
    now: datetime,
    classifier_version: str,
    skills_dir: Path,
) -> ReportData:
    items, distribution = load_report_data(
        engine, since=since, classifier_version=classifier_version, skills_dir=skills_dir
    )
    patterns = with_proposals(engine, proposal_call, group_gists(call, items))
    save_patterns(engine, patterns, now=now)
    return ReportData(since=since, distribution=distribution, patterns=tuple(patterns))


def recheck_count(pattern: Pattern) -> int:
    """Эпизоды, чья суть мало похожа на название паттерна: кандидаты в «фальшивые».

    Это прокси по сходству формулировок, не оценка модели (см. ADR 0009).
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
        if p.proposal is not None:
            lines += render_proposal(p.proposal)
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
    """Путь вывода годится для записи; зовётся до модели и до сохранения паттернов."""
    if path.name.casefold() in PROTECTED_NAMES:
        raise InvalidReportPathError(path)
    if path.is_dir():
        raise ReportWriteError(path, "это каталог")
    parent = path.parent
    if not parent.is_dir():
        raise ReportWriteError(path, f"каталога {parent} нет")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise ReportWriteError(path, f"нет прав на запись в {parent}")
    if path.exists() and not os.access(path, os.W_OK):
        raise ReportWriteError(path, "нет прав на запись в файл")


def write_report(path: Path, text: str) -> None:
    check_report_path(path)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ReportWriteError(path, exc.strerror or type(exc).__name__) from exc


def run_report(
    engine: Engine,
    call: ModelCall,
    proposal_call: ModelCall,
    *,
    output: Path,
    since: datetime | None,
    now: datetime,
    classifier_version: str,
    skills_dir: Path,
) -> ReportData:
    """Проверка пути, паттерны, отчёт. Неверный путь не стоит ни вызова модели, ни паттернов."""
    check_report_path(output)
    data = build_report(
        engine,
        call,
        proposal_call,
        since=since,
        now=now,
        classifier_version=classifier_version,
        skills_dir=skills_dir,
    )
    write_report(output, render_report(data))
    return data
