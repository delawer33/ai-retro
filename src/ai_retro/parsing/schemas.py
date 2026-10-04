from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Record:
    """Строка лога, сведённая к нужным полям. Выход стыка «файлы → записи»."""

    uuid: str
    parent_uuid: str | None
    session_id: str
    project: str
    timestamp: datetime
    kind: str  # "user" | "assistant" | другое
    text: str | None  # то, что написал пользователь; None для всего остального
    source: Path
    agent_text: str | None = None  # текст ответа агента; None для не-assistant и без текста


@dataclass(frozen=True)
class Rewind:
    """Точка rewind'а: первая реплика новой ветки и начало брошенной ветки."""

    episode_uuid: str
    abandoned: tuple[Record, ...]  # исходная реплика и первый ответ агента на неё


@dataclass(frozen=True)
class Conversation:
    """Выход стыка «записи → беседы»: записи без дублей по uuid со всех файлов беседы."""

    id: str
    records: tuple[Record, ...]
    rewinds: tuple[Rewind, ...] = ()


@dataclass(frozen=True)
class EpisodeDraft:
    """Выход стыка «записи/беседы → эпизоды»."""

    uuid: str
    conversation_id: str
    project: str
    timestamp: datetime
    text: str
    context: str | None
    signals: tuple[str, ...]
