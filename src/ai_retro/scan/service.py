import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from ai_retro.classify.service import (
    LOOP_MARKER,
    EpisodeInput,
    build_prompt,
    classify_many,
)
from ai_retro.db.models import (
    ClassificationRow,
    EpisodeRow,
    GoldLabelRow,
    GoldReturnedRow,
    GoldSampleRow,
)
from ai_retro.domain.service import OTHER, assign_domains, is_scratch
from ai_retro.errors import InvalidSinceError, LogsDirNotFoundError
from ai_retro.llm import ModelCall
from ai_retro.parsing.conversations import records_to_conversations
from ai_retro.parsing.episodes import conversations_to_episodes, skill_arguments
from ai_retro.parsing.files import files_to_records, find_log_files
from ai_retro.parsing.schemas import EpisodeDraft

BATCH_SIZE = 20
_UNITS = {"h": timedelta(hours=1), "d": timedelta(days=1), "w": timedelta(weeks=1)}


@dataclass(frozen=True)
class Review:
    """Итог пересмотра сохранённых эпизодов по текущим правилам фильтра (ADR 0011)."""

    removed: int = 0
    updated: int = 0
    removed_gold_labels: int = 0


@dataclass(frozen=True)
class ScanResult:
    found: int
    classified: int
    unclassified: int = 0
    review: Review = Review()


def parse_since(value: str, *, now: datetime) -> datetime:
    match = re.fullmatch(r"(\d+)([hdw])", value.strip())
    if match is None:
        raise InvalidSinceError(value)
    return now - int(match.group(1)) * _UNITS[match.group(2)]


def scan(
    engine: Engine,
    call: ModelCall,
    *,
    logs_dir: Path,
    since: datetime | None,
    threads: int,
    classifier_version: str,
    context_chars: int = 2000,
    announce: Callable[[int], None] | None = None,
    domain_rules: list[tuple[Path, str]] | None = None,
    domain_call: ModelCall | None = None,
    skills_dir: Path,
    reviewed: Callable[[Review], None] | None = None,
) -> ScanResult:
    if not logs_dir.is_dir():
        raise LogsDirNotFoundError(logs_dir)
    records = files_to_records(find_log_files(logs_dir))
    conversations = [
        c
        for c in records_to_conversations(records)
        if not any(r.text and LOOP_MARKER in r.text for r in c.records)
    ]
    all_drafts = conversations_to_episodes(conversations, skills_dir=skills_dir)
    drafts = [d for d in all_drafts if since is None or d.timestamp.astimezone(UTC) >= since]
    window_ids = {
        c.id
        for c in conversations
        if since is None or any(r.timestamp.astimezone(UTC) >= since for r in c.records)
    }

    with Session(engine) as session:
        review = review_stored(
            session, [d for d in all_drafts if d.conversation_id in window_ids], window_ids
        )
        session.commit()
        if reviewed is not None:
            reviewed(review)
        known: dict[str, str] = {}
        by_model: set[str] = set()
        for conversation_id, project, domain, decided in session.execute(
            select(
                EpisodeRow.conversation_id,
                EpisodeRow.project,
                EpisodeRow.domain,
                EpisodeRow.domain_by_model,
            )
            .where(EpisodeRow.domain.is_not(None))
            .distinct()
        ):
            if domain is None:
                continue
            # scratch с «иное» без решения модели — сбой или прежний код: пересматриваем
            if domain_call is not None and domain == OTHER and is_scratch(project) and not decided:
                continue
            known[conversation_id] = domain
            if decided:
                by_model.add(conversation_id)
        done = set(
            session.scalars(
                select(ClassificationRow.episode_uuid).where(
                    ClassificationRow.classifier_version == classifier_version
                )
            )
        )
        todo = [d for d in drafts if d.uuid not in done and d.category is None]
        in_window = {d.conversation_id for d in drafts}
        fresh = [c for c in conversations if c.id in in_window and c.id not in known]
        assigned = assign_domains(
            fresh,
            domain_rules or [],
            domain_call,
            threads=threads,
            before_calls=(lambda n: announce(len(todo) + n)) if announce is not None else None,
        )
        domains: dict[str, str | None] = {**known, **assigned.domains}
        by_model |= assigned.by_model
        for draft in drafts:
            session.merge(
                EpisodeRow(
                    uuid=draft.uuid,
                    conversation_id=draft.conversation_id,
                    project=draft.project,
                    domain=domains.get(draft.conversation_id),
                    domain_by_model=draft.conversation_id in by_model,
                    timestamp=draft.timestamp,
                    text=draft.text,
                    context=draft.context,
                    signals=list(draft.signals),
                    skill=draft.skill,
                )
            )
        session.commit()

        for draft in drafts:
            if draft.category is not None and draft.uuid not in done:
                # Категория известна без модели: голый вызов скилла.
                session.add(
                    ClassificationRow(
                        episode_uuid=draft.uuid,
                        category=draft.category,
                        gist="",
                        classifier_version=classifier_version,
                    )
                )
        session.commit()
        classified = 0
        unclassified = 0
        for start in range(0, len(todo), BATCH_SIZE):
            batch = [
                (d.uuid, build_prompt(_episode_input(d), context_chars=context_chars))
                for d in todo[start : start + BATCH_SIZE]
            ]
            outcome = classify_many(call, batch, threads=threads)
            for uuid, result in outcome.results.items():
                session.add(
                    ClassificationRow(
                        episode_uuid=uuid,
                        category=result.category,
                        gist=result.gist,
                        classifier_version=classifier_version,
                    )
                )
                classified += 1
            unclassified += len(outcome.unclassified)
            session.commit()
            if outcome.error is not None:
                raise outcome.error
    return ScanResult(
        found=len(drafts), classified=classified, unclassified=unclassified, review=review
    )


def review_stored(
    session: Session, current: list[EpisodeDraft], conversation_ids: set[str]
) -> Review:
    """Сверить эпизоды в базе с пересобранными по текущим правилам.

    Лишние удаляются вместе с классификациями и строками эталона; у остальных обновляются
    текст и признаки (тот же uuid). Классификации оставшихся не трогаются.
    """
    by_uuid = {d.uuid: d for d in current}
    gone: list[str] = []
    updated = 0
    for row in session.scalars(
        select(EpisodeRow).where(EpisodeRow.conversation_id.in_(conversation_ids))
    ):
        draft = by_uuid.get(row.uuid)
        if draft is None:
            gone.append(row.uuid)
            continue
        fresh = {
            "project": draft.project,
            "text": draft.text,
            "context": draft.context,
            "signals": list(draft.signals),
            "skill": draft.skill,
        }
        if any(getattr(row, name) != value for name, value in fresh.items()):
            for name, value in fresh.items():
                setattr(row, name, value)
            updated += 1
    if not gone:
        return Review(updated=updated)
    labels = len(
        session.scalars(select(GoldLabelRow.id).where(GoldLabelRow.episode_uuid.in_(gone))).all()
    )
    for table in (ClassificationRow, GoldLabelRow, GoldReturnedRow, GoldSampleRow):
        session.execute(delete(table).where(table.episode_uuid.in_(gone)))
    session.execute(delete(EpisodeRow).where(EpisodeRow.uuid.in_(gone)))
    return Review(removed=len(gone), updated=updated, removed_gold_labels=labels)


def _episode_input(draft: EpisodeDraft) -> EpisodeInput:
    # Для rewind joint «записи → эпизоды» кладёт в context начало брошенной ветки,
    # иначе — последний ответ агента перед репликой.
    rewind = "rewind" in draft.signals
    return EpisodeInput(
        # У вызова скилла в модель идут только аргументы: команда — в признаке.
        text=skill_arguments(draft.text) if draft.skill else draft.text,
        signals=draft.signals,
        agent_reply=None if rewind else draft.context,
        abandoned_branch=draft.context if rewind else None,
    )
