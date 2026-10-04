"""Стык «записи → беседы» (ADR 0001): беседа собирается из всех файлов проекта по общим uuid."""

from collections import defaultdict
from collections.abc import Iterable

from ai_retro.parsing.schemas import Conversation, Record, Rewind


def records_to_conversations(records: Iterable[Record]) -> list[Conversation]:
    files = _group_files(records)
    groups = _connected_files(files)
    return [_assemble(group, files) for group in groups]


def _group_files(records: Iterable[Record]) -> dict[str, list[Record]]:
    """Файлы в порядке начала (по самой ранней записи); записи внутри файла в порядке строк."""
    by_file: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_file[str(record.source)].append(record)
    # Копия истории сохраняет метки времени: при равном начале раньше файл, что кончился раньше.
    ordered = sorted(
        by_file.items(),
        key=lambda kv: (
            min(r.timestamp for r in kv[1]),
            max(r.timestamp for r in kv[1]),
            kv[0],
        ),
    )
    return dict(ordered)


def _connected_files(files: dict[str, list[Record]]) -> list[list[str]]:
    """Файлы, связанные общими uuid, — одна беседа. Порядок файлов сохраняется."""
    parent = {path: path for path in files}

    def find(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    first_seen: dict[str, str] = {}
    for path, recs in files.items():
        for r in recs:
            other = first_seen.setdefault(r.uuid, path)
            parent[find(path)] = find(other)
    groups: dict[str, list[str]] = defaultdict(list)
    for path in files:
        groups[find(path)].append(path)
    return list(groups.values())


def _assemble(paths: list[str], files: dict[str, list[Record]]) -> Conversation:
    seen: set[str] = set()
    records: list[Record] = []
    rewinds: list[Rewind] = []
    earlier: list[list[Record]] = []
    for path in paths:
        recs = files[path]
        rewind = _detect_rewind(recs, earlier, seen)
        if rewind is not None:
            rewinds.append(rewind)
        for r in recs:
            if r.uuid not in seen:
                seen.add(r.uuid)
                records.append(r)
        earlier.append(recs)
    return Conversation(id=records[0].session_id, records=tuple(records), rewinds=tuple(rewinds))


def _detect_rewind(
    recs: list[Record], earlier: list[list[Record]], seen: set[str]
) -> Rewind | None:
    """Файл — rewind, если делит префикс с более ранним и расходится с ним.

    Брошенная ветка должна содержать реплику пользователя, которой нет в новом файле: хвост из
    параллельных tool calls не в счёт, а `resume` (новый файл содержит старый целиком) — не rewind.
    """
    own = [r for r in recs if r.uuid not in seen]
    first_own = next((r for r in own if r.kind == "user" and r.text is not None), None)
    if first_own is None or len(own) == len(recs):
        return None
    uuids = {r.uuid for r in recs}
    best: tuple[int, list[Record]] | None = None
    for old in earlier:
        shared = sum(1 for r in old if r.uuid in uuids)
        abandoned = [r for r in old if r.uuid not in uuids]
        if shared == 0 or not any(r.kind == "user" and r.text is not None for r in abandoned):
            continue
        if best is None or shared >= best[0]:
            best = (shared, abandoned)
    if best is None:
        return None
    return Rewind(episode_uuid=first_own.uuid, abandoned=_branch_start(best[1]))


def _branch_start(abandoned: list[Record]) -> tuple[Record, ...]:
    """Исходная реплика брошенной ветки и первый ответ агента с текстом."""
    start = next(i for i, r in enumerate(abandoned) if r.kind == "user" and r.text is not None)
    reply = None
    for r in abandoned[start + 1 :]:
        if r.kind == "user" and r.text is not None:
            break
        if r.kind == "assistant" and r.agent_text is not None:
            reply = r
            break
    return (abandoned[start],) if reply is None else (abandoned[start], reply)
