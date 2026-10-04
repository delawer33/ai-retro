"""Стык «записи/беседы → эпизоды»."""

from collections.abc import Iterable

from ai_retro.parsing.schemas import Conversation, EpisodeDraft, Record, Rewind

REWIND = "rewind"
_CONTEXT_LIMIT = 1000


def conversations_to_episodes(conversations: Iterable[Conversation]) -> list[EpisodeDraft]:
    """Эпизод = текстовая реплика пользователя; у первой реплики после rewind — признак и контекст."""
    return [
        EpisodeDraft(
            uuid=record.uuid,
            conversation_id=conversation.id,
            project=record.project,
            timestamp=record.timestamp,
            text=record.text,
            context=_rewind_context(rewinds[record.uuid]) if record.uuid in rewinds else None,
            signals=(REWIND,) if record.uuid in rewinds else (),
        )
        for conversation in conversations
        for rewinds in [{r.episode_uuid: r for r in conversation.rewinds}]
        for record in conversation.records
        if record.kind == "user" and record.text is not None
    ]


def _rewind_context(rewind: Rewind) -> str:
    lines = ["Брошенная ветка (начало):"]
    for record in rewind.abandoned:
        who, text = (
            ("Пользователь", record.text)
            if record.kind == "user"
            else (
                "Агент",
                record.agent_text,
            )
        )
        lines.append(f"{who}: {_clip(text)}")
    return "\n".join(lines)


def _clip(text: str | None) -> str:
    text = text or ""
    return text if len(text) <= _CONTEXT_LIMIT else text[:_CONTEXT_LIMIT] + "…"


__all__ = ["REWIND", "Record", "conversations_to_episodes"]
