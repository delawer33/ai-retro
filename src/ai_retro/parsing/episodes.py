"""Стык «записи/беседы → эпизоды»."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ai_retro.parsing.schemas import Conversation, EpisodeDraft, Record, Rewind

REWIND = "rewind"
SIGNAL_INTERRUPT = "прерывание"
SIGNAL_REJECTION = "отказ"
SIGNAL_SKILL = "вызов скилла"
NEUTRAL = "нейтрально"
# Вариант «… for tool use» — эхо отказа; сам по себе не реплика и не признак.
INTERRUPT_MARKER = "[Request interrupted by user]"
_CONTEXT_LIMIT = 1000

# Записи, целиком подмешанные харнессом: не реплика пользователя.
_NOT_REPLY_PREFIXES = (
    "<task-notification",
    "<local-command-",
    "<bash-input",
    "<bash-stdout",
    "<bash-stderr",
    "This session is being continued from a previous conversation",
)
_SYSTEM_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_COMMAND = re.compile(
    r"(?:<command-message>.*?</command-message>\s*)?<command-name>(.*?)</command-name>", re.DOTALL
)
_COMMAND_ARGS = re.compile(r"<command-args>(.*?)</command-args>", re.DOTALL)
_PASTE = re.compile(r"<pasted_content[^>]*>(.*?)</pasted_content[^>]*>", re.DOTALL)


@dataclass(frozen=True)
class CleanedReply:
    text: str | None  # то, что написал пользователь; None — реплики нет
    is_command: bool = False  # text — «/имя аргументы»
    is_interrupt: bool = False  # текст — маркер прерывания, не реплика


def clean_reply(text: str) -> CleanedReply:
    """Очистить текст реплики от подмешанного харнессом.

    `<system-reminder>` убирается блоком; `<command-*>` становится «/имя аргументы»;
    `<pasted_content>` — «[вставка N символов]». Уведомления задач, вывод локальных команд,
    compact-сводка и маркер прерывания реплики не дают.
    """
    stripped = _SYSTEM_REMINDER.sub("", text).strip()
    if not stripped:
        return CleanedReply(None)
    if stripped == INTERRUPT_MARKER:
        return CleanedReply(None, is_interrupt=True)
    if stripped.startswith(_NOT_REPLY_PREFIXES) or stripped.startswith(INTERRUPT_MARKER[:-1]):
        return CleanedReply(None)
    command = _COMMAND.match(stripped)
    if command is not None:
        args = _COMMAND_ARGS.search(stripped)
        parts = [command.group(1).strip(), args.group(1).strip() if args else ""]
        return CleanedReply(_mask_pastes(" ".join(p for p in parts if p)), is_command=True)
    return CleanedReply(_mask_pastes(stripped))


def _mask_pastes(text: str) -> str:
    return _PASTE.sub(lambda m: f"[вставка {len(m.group(1))} символов]", text)


def conversations_to_episodes(conversations: Iterable[Conversation]) -> list[EpisodeDraft]:
    """Эпизод — реплика пользователя; прерывание и отказ без текста уходят признаком в следующий.

    У первой реплики после rewind — признак и контекст (начало брошенной ветки), иначе
    контекст — последний ответ агента.
    """
    drafts: list[EpisodeDraft] = []
    for conversation in conversations:
        rewinds = {r.episode_uuid: r for r in conversation.rewinds}
        last_reply: str | None = None
        pending: list[str] = []
        for record in conversation.records:
            if record.kind == "assistant" and record.agent_text:
                last_reply = record.agent_text
            if record.is_interrupt:
                _add(pending, SIGNAL_INTERRUPT)
            if record.is_rejection:
                _add(pending, SIGNAL_REJECTION)
            if record.kind != "user" or record.text is None:
                continue
            rewind = rewinds.get(record.uuid)
            if rewind is not None:
                _add(pending, REWIND)
            skill: str | None = None
            category: str | None = None
            if record.is_command:
                skill = record.text.split(maxsplit=1)[0].lstrip("/")
                _add(pending, SIGNAL_SKILL)
                if not skill_arguments(record.text):
                    category = NEUTRAL  # голый вызов в модель не идёт (ADR 0004)
            drafts.append(
                EpisodeDraft(
                    uuid=record.uuid,
                    conversation_id=conversation.id,
                    project=record.project,
                    timestamp=record.timestamp,
                    text=record.text,
                    context=_rewind_context(rewind) if rewind else last_reply,
                    signals=tuple(pending),
                    category=category,
                    skill=skill,
                )
            )
            pending = []
    return drafts


def skill_arguments(text: str) -> str:
    """Аргументы слэш-команды из текста эпизода «/имя аргументы»; пусто — голый вызов."""
    parts = text.split(maxsplit=1)
    return parts[1] if len(parts) == 2 else ""


def _add(signals: list[str], signal: str) -> None:
    if signal not in signals:
        signals.append(signal)


def _rewind_context(rewind: Rewind) -> str:
    lines = ["Брошенная ветка (начало):"]
    for record in rewind.abandoned:
        who, text = (
            ("Пользователь", record.text) if record.kind == "user" else ("Агент", record.agent_text)
        )
        lines.append(f"{who}: {_clip(text)}")
    return "\n".join(lines)


def _clip(text: str | None) -> str:
    text = text or ""
    return text if len(text) <= _CONTEXT_LIMIT else text[:_CONTEXT_LIMIT] + "…"


__all__ = [
    "NEUTRAL",
    "REWIND",
    "SIGNAL_SKILL",
    "CleanedReply",
    "Record",
    "clean_reply",
    "conversations_to_episodes",
    "skill_arguments",
]
