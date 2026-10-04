"""Стык «записи/беседы → эпизоды»."""

from collections.abc import Iterable

from ai_retro.parsing.schemas import Conversation, EpisodeDraft


def conversations_to_episodes(conversations: Iterable[Conversation]) -> list[EpisodeDraft]:
    """Пока эпизод = текстовая реплика пользователя; контекст — последний текстовый ответ
    агента перед репликой; признаки — отдельный тикет."""
    drafts: list[EpisodeDraft] = []
    for conversation in conversations:
        last_reply: str | None = None
        for record in conversation.records:
            if record.kind == "assistant" and record.assistant_text:
                last_reply = record.assistant_text
            elif record.kind == "user" and record.text is not None:
                drafts.append(
                    EpisodeDraft(
                        uuid=record.uuid,
                        conversation_id=conversation.id,
                        project=record.project,
                        timestamp=record.timestamp,
                        text=record.text,
                        context=last_reply,
                        signals=(),
                    )
                )
    return drafts
