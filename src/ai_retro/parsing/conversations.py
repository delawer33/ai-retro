"""Стык «записи → беседы»."""

from collections import defaultdict
from collections.abc import Iterable

from ai_retro.parsing.schemas import Conversation, Record


def records_to_conversations(records: Iterable[Record]) -> list[Conversation]:
    """Пока один файл = одна беседа; сборка из нескольких файлов и rewind — отдельный тикет."""
    by_file: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_file[str(record.source)].append(record)
    return [Conversation(id=rs[0].session_id, records=tuple(rs)) for rs in by_file.values()]
