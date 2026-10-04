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
    assistant_text: str | None = None  # текстовые блоки ответа агента


@dataclass(frozen=True)
class Conversation:
    """Выход стыка «записи → беседы»."""

    id: str
    records: tuple[Record, ...]


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
