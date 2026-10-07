"""Эталон: стратифицированная выборка, разбиение dev/test, очередь разметки."""

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from ai_retro.classify.service import CATEGORIES
from ai_retro.db.models import (
    ClassificationRow,
    EpisodeRow,
    GoldLabelRow,
    GoldReturnedRow,
    GoldSampleRow,
)
from ai_retro.errors import LabelExistsError
from ai_retro.parsing.episodes import SIGNAL_SKILL

SAMPLE_SIZE = 100
RELABEL_SIZE = 20
MAX_PER_CONVERSATION = 3
TEST_SHARE = 0.3
# Доли страт: с признаками / модель сказала ненейтрально / модель сказала нейтрально.
STRATA_SHARES = (0.4, 0.4, 0.2)
NEUTRAL = "нейтрально"


@dataclass(frozen=True)
class Candidate:
    uuid: str
    conversation_id: str
    project: str
    has_signals: bool
    model_category: str | None  # только для страты; на разметке не показывается


@dataclass(frozen=True)
class LabelItem:
    """Всё, что видит человек при разметке. Метки модели здесь нет намеренно."""

    uuid: str
    text: str
    context: str | None
    signals: tuple[str, ...]
    skill: str | None = None


def _order_key(uuid: str) -> str:
    return hashlib.sha256(uuid.encode()).hexdigest()


def _stratum(c: Candidate) -> int | None:
    if c.has_signals:
        return 0
    if c.model_category is None:
        return None
    return 2 if c.model_category == NEUTRAL else 1


def _pick(
    candidates: Sequence[Candidate], quota: int, per_conversation: dict[str, int]
) -> list[Candidate]:
    """По кругу по проектам, ≤ MAX_PER_CONVERSATION на беседу (счётчик общий для всех страт);
    порядок внутри проекта — детерминированный хеш."""
    by_project: dict[str, list[Candidate]] = defaultdict(list)
    for c in sorted(candidates, key=lambda c: _order_key(c.uuid)):
        by_project[c.project].append(c)
    projects = sorted(by_project, key=_order_key)
    picked: list[Candidate] = []
    while len(picked) < quota and any(by_project.values()):
        for project in projects:
            queue = by_project[project]
            while queue and per_conversation[queue[0].conversation_id] >= MAX_PER_CONVERSATION:
                queue.pop(0)
            if queue and len(picked) < quota:
                c = queue.pop(0)
                per_conversation[c.conversation_id] += 1
                picked.append(c)
    return picked


def draw_sample(
    candidates: Sequence[Candidate],
    *,
    size: int = SAMPLE_SIZE,
    taken_per_conversation: dict[str, int] | None = None,
) -> list[tuple[Candidate, str]]:
    """Выборка по стратам и разбиение на dev/test. Возвращает (кандидат, split).

    Страт не хватает — берём сколько есть. Разбиение: по хешу uuid, первые 30% — test.
    """
    strata: list[list[Candidate]] = [[], [], []]
    for c in candidates:
        index = _stratum(c)
        if index is not None:
            strata[index].append(c)
    per_conversation: dict[str, int] = defaultdict(int, taken_per_conversation or {})
    picked: list[Candidate] = []
    for stratum, share in zip(strata, STRATA_SHARES, strict=True):
        picked += _pick(stratum, round(size * share), per_conversation)
    ranked = sorted(picked, key=lambda c: _order_key(c.uuid))
    test_uuids = {c.uuid for c in ranked[: round(len(ranked) * TEST_SHARE)]}
    return [(c, "test" if c.uuid in test_uuids else "dev") for c in picked]


def load_candidates(engine: Engine) -> list[Candidate]:
    with Session(engine) as session:
        latest: dict[str, str] = {}
        for row in session.scalars(select(ClassificationRow).order_by(ClassificationRow.id)):
            latest[row.episode_uuid] = row.category
        return [
            Candidate(
                uuid=e.uuid,
                conversation_id=e.conversation_id,
                project=e.project,
                has_signals=bool(e.signals),
                model_category=latest.get(e.uuid),
            )
            for e in session.scalars(select(EpisodeRow))
        ]


def ensure_sample(engine: Engine, *, size: int = SAMPLE_SIZE, extend: bool = False) -> int:
    """Сохраняет выборку, если её ещё нет (или при extend — добавляет новую). Возвращает, сколько
    эпизодов добавлено."""
    with Session(engine) as session:
        sampled = list(session.scalars(select(GoldSampleRow.episode_uuid)))
        if sampled and not extend:
            return 0
        sampled_set = set(sampled)
        candidates = [c for c in load_candidates(engine) if c.uuid not in sampled_set]
        taken: dict[str, int] = defaultdict(int)
        for conversation_id in session.scalars(
            select(EpisodeRow.conversation_id).where(EpisodeRow.uuid.in_(sampled_set))
        ):
            taken[conversation_id] += 1
        drawn = draw_sample(candidates, size=size, taken_per_conversation=dict(taken))
        for c, split in drawn:
            session.add(GoldSampleRow(episode_uuid=c.uuid, split=split))
        session.commit()
        return len(drawn)


def _item(episode: EpisodeRow) -> LabelItem:
    return LabelItem(
        uuid=episode.uuid,
        text=episode.text,
        context=episode.context,
        signals=tuple(episode.signals),
        skill=episode.skill,
    )


def pending_items(engine: Engine) -> list[LabelItem]:
    """Размеченные пропускаются; продолжение — с первого неразмеченного в порядке выборки."""
    with Session(engine) as session:
        done = set(
            session.scalars(
                select(GoldLabelRow.episode_uuid).where(GoldLabelRow.is_relabel.is_(False))
            )
        )
        rows = session.execute(
            select(GoldSampleRow, EpisodeRow)
            .join(EpisodeRow, EpisodeRow.uuid == GoldSampleRow.episode_uuid)
            .order_by(GoldSampleRow.id)
        ).all()
        return [_item(e) for s, e in rows if s.episode_uuid not in done]


def returned_count(engine: Engine) -> int:
    """Сколько эпизодов вернула в разметку миграция эталона (ADR 0004) и они ещё не размечены."""
    with Session(engine) as session:
        done = set(
            session.scalars(
                select(GoldLabelRow.episode_uuid).where(GoldLabelRow.is_relabel.is_(False))
            )
        )
        return sum(
            1 for u in session.scalars(select(GoldReturnedRow.episode_uuid)) if u not in done
        )


def pending_relabel_items(engine: Engine, *, size: int = RELABEL_SIZE) -> list[LabelItem]:
    """Повторная разметка вслепую: детерминированные `size` размеченных эпизодов."""
    with Session(engine) as session:
        labeled = list(
            session.scalars(
                select(GoldLabelRow.episode_uuid).where(GoldLabelRow.is_relabel.is_(False))
            )
        )
        relabeled = set(
            session.scalars(
                select(GoldLabelRow.episode_uuid).where(GoldLabelRow.is_relabel.is_(True))
            )
        )
        chosen = sorted(labeled, key=_order_key)[:size]
        todo = [u for u in chosen if u not in relabeled]
        episodes = {
            e.uuid: e for e in session.scalars(select(EpisodeRow).where(EpisodeRow.uuid.in_(todo)))
        }
        return [_item(episodes[u]) for u in todo]


def save_label(engine: Engine, uuid: str, category: str, *, relabel: bool, now: datetime) -> None:
    """Добавляет метку. Исходная метка никогда не переписывается."""
    if category not in CATEGORIES:
        raise ValueError(f"Неизвестная категория: {category!r}")
    with Session(engine) as session:
        original = session.scalars(
            select(GoldLabelRow).where(
                GoldLabelRow.episode_uuid == uuid, GoldLabelRow.is_relabel.is_(False)
            )
        ).first()
        if relabel:
            if original is None:
                raise LabelExistsError(uuid)  # повторять нечего; не должно случаться из CLI
            split = original.split
        else:
            if original is not None:
                raise LabelExistsError(uuid)
            split = session.scalars(
                select(GoldSampleRow.split).where(GoldSampleRow.episode_uuid == uuid)
            ).one()
        session.add(
            GoldLabelRow(
                episode_uuid=uuid,
                category=category,
                split=split,
                labeled_at=now,
                is_relabel=relabel,
            )
        )
        session.commit()


def render_item(item: LabelItem, *, number: int, total: int) -> str:
    lines = [f"--- Эпизод {number}/{total} ---"]
    if item.signals:
        shown = [
            f"{s} /{item.skill}" if s == SIGNAL_SKILL and item.skill else s for s in item.signals
        ]
        lines.append("Признаки: " + ", ".join(shown))
    if item.context:
        lines.append("Контекст (ответ агента / начало брошенной ветки):")
        lines.append(item.context)
    lines.append("Реплика:")
    lines.append(item.text)
    return "\n".join(lines)
