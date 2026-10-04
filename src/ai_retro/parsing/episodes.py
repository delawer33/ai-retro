"""Стык «записи/беседы → эпизоды»."""

from collections.abc import Iterable

from ai_retro.parsing.schemas import Conversation, EpisodeDraft


def conversations_to_episodes(conversations: Iterable[Conversation]) -> list[EpisodeDraft]:
    """Пока эпизод = текстовая реплика пользователя; признаки и контекст — отдельный тикет."""
    return [
        EpisodeDraft(
            uuid=record.uuid,
            conversation_id=conversation.id,
            project=record.project,
            timestamp=record.timestamp,
            text=record.text,
            context=None,
            signals=(),
        )
        for conversation in conversations
        for record in conversation.records
        if record.kind == "user" and record.text is not None
    ]
