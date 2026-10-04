"""Стык «записи/беседы → эпизоды»."""

from collections.abc import Iterable

from ai_retro.parsing.schemas import Conversation, EpisodeDraft, Record, Rewind

REWIND = "rewind"
_CONTEXT_LIMIT = 1000


def conversations_to_episodes(conversations: Iterable[Conversation]) -> list[EpisodeDraft]:
    """Эпизод = текстовая реплика пользователя; у первой реплики после rewind — признак и контекст."""
    drafts: list[EpisodeDraft] = []
    for conversation in conversations:
        rewinds = {r.episode_uuid: r for r in conversation.rewinds}
        last_reply: str | None = None
        for record in conversation.records:
            if record.kind == "assistant" and record.agent_text:
                last_reply = record.agent_text
            elif record.kind == "user" and record.text is not None:
                rewind = rewinds.get(record.uuid)
                drafts.append(
                    EpisodeDraft(
                        uuid=record.uuid,
                        conversation_id=conversation.id,
                        project=record.project,
                        timestamp=record.timestamp,
                        text=record.text,
                        context=_rewind_context(rewind) if rewind else last_reply,
                        signals=(REWIND,) if rewind else (),
                    )
                )
    return drafts


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
