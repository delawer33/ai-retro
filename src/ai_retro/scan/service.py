import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from ai_retro.classify.service import CLASSIFIER_VERSION, LOOP_MARKER, classify_many
from ai_retro.db.models import ClassificationRow, EpisodeRow
from ai_retro.domain.service import OTHER, assign_domains, is_scratch
from ai_retro.errors import InvalidSinceError, LogsDirNotFoundError
from ai_retro.llm import ModelCall
from ai_retro.parsing.conversations import records_to_conversations
from ai_retro.parsing.episodes import conversations_to_episodes
from ai_retro.parsing.files import files_to_records, find_log_files

BATCH_SIZE = 20
_UNITS = {"h": timedelta(hours=1), "d": timedelta(days=1), "w": timedelta(weeks=1)}


@dataclass(frozen=True)
class ScanResult:
    found: int
    classified: int


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
    domain_rules: list[tuple[Path, str]] | None = None,
    domain_call: ModelCall | None = None,
) -> ScanResult:
    if not logs_dir.is_dir():
        raise LogsDirNotFoundError(logs_dir)
    records = files_to_records(find_log_files(logs_dir))
    conversations = [
        c
        for c in records_to_conversations(records)
        if not any(r.text and LOOP_MARKER in r.text for r in c.records)
    ]
    drafts = [
        d
        for d in conversations_to_episodes(conversations)
        if since is None or d.timestamp.astimezone(UTC) >= since
    ]

    with Session(engine) as session:
        known = {
            conversation_id: domain
            for conversation_id, project, domain in session.execute(
                select(EpisodeRow.conversation_id, EpisodeRow.project, EpisodeRow.domain)
                .where(EpisodeRow.domain.is_not(None))
                .distinct()
            )
            # scratch с «иное» — следствие сбоя или прежнего кода: пересматриваем, если есть модель
            if not (domain_call is not None and domain == OTHER and is_scratch(project))
        }
        in_window = {d.conversation_id for d in drafts}
        fresh = [c for c in conversations if c.id in in_window and c.id not in known]
        domains: dict[str, str | None] = {**known}
        domains |= assign_domains(fresh, domain_rules or [], domain_call, threads=threads)
        for draft in drafts:
            session.merge(
                EpisodeRow(
                    uuid=draft.uuid,
                    conversation_id=draft.conversation_id,
                    project=draft.project,
                    domain=domains.get(draft.conversation_id),
                    timestamp=draft.timestamp,
                    text=draft.text,
                    context=draft.context,
                    signals=list(draft.signals),
                )
            )
        session.commit()

        done = set(
            session.scalars(
                select(ClassificationRow.episode_uuid).where(
                    ClassificationRow.classifier_version == CLASSIFIER_VERSION
                )
            )
        )
        todo = [(d.uuid, d.text) for d in drafts if d.uuid not in done]
        classified = 0
        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start : start + BATCH_SIZE]
            for uuid, result in classify_many(call, batch, threads=threads).items():
                session.add(
                    ClassificationRow(
                        episode_uuid=uuid,
                        category=result.category,
                        gist=result.gist,
                        classifier_version=CLASSIFIER_VERSION,
                    )
                )
                classified += 1
            session.commit()
    return ScanResult(found=len(drafts), classified=classified)
