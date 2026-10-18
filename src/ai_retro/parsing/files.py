"""Стык «файлы → записи»."""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_retro.parsing.episodes import clean_reply
from ai_retro.parsing.schemas import Record

_REJECTION_PREFIX = "The user doesn't want to"
_REJECTION_REASON = "the user said:"
_WORKTREE_CWD = re.compile(r"/\.claude/worktrees/.*$")
_WORKTREE_DIR = re.compile(r"--claude-worktrees-.*$")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


def find_log_files(logs_dir: Path) -> list[Path]:
    return sorted(logs_dir.glob("*/*.jsonl"))


def files_to_records(files: Iterable[Path]) -> list[Record]:
    """Записи без cwd получают проект по имени каталога лога (см. `_project_by_dir`)."""
    parsed: list[tuple[Record, bool]] = []
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                item = _parse_line(line, path)
                if item is not None:
                    parsed.append(item)
    known = {_encode(r.project): r.project for r, has_cwd in parsed if has_cwd}
    return [
        r if has_cwd else replace(r, project=_project_by_dir(r.source.parent.name, known))
        for r, has_cwd in parsed
    ]


def _parse_line(line: str, source: Path) -> tuple[Record, bool] | None:
    """Запись и признак, что её проект взят из cwd (иначе — заглушка до `files_to_records`)."""
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    kind = raw.get("type")
    uuid = raw.get("uuid")
    timestamp = raw.get("timestamp")
    if kind not in ("user", "assistant") or not uuid or not timestamp:
        return None
    parsed = _parse_user(raw) if kind == "user" else _UserParts()
    cwd = raw.get("cwd")
    record = Record(
        uuid=uuid,
        parent_uuid=raw.get("parentUuid"),
        session_id=raw.get("sessionId", source.stem),
        project=parent_project(cwd) if cwd else "",
        timestamp=datetime.fromisoformat(timestamp),
        kind=kind,
        text=parsed.text,
        source=source,
        agent_text=_agent_text(raw) if kind == "assistant" else None,
        is_command=parsed.is_command,
        is_interrupt=parsed.is_interrupt,
        is_rejection=parsed.is_rejection,
    )
    return record, bool(cwd)


def parent_project(cwd: str) -> str:
    """Проект — путь родительского репозитория: каталог worktree относится к нему."""
    return _WORKTREE_CWD.sub("", cwd)


def _encode(path: str) -> str:
    """Имя каталога лога для пути: всё, кроме букв и цифр, становится «-»."""
    return _NON_ALNUM.sub("-", path)


def _project_by_dir(dir_name: str, known: dict[str, str]) -> str:
    """Проект по имени каталога лога — в той же форме, что и из cwd.

    Имя — dash-кодировка пути и без потерь не декодируется, поэтому сначала ищем путь среди cwd
    других записей (в том числе родительского проекта для каталога worktree), и только потом
    грубо заменяем «-» на «/». См. ADR 0005.
    """
    key = _WORKTREE_DIR.sub("", dir_name)
    return known.get(key) or key.replace("-", "/")


@dataclass(frozen=True)
class _UserParts:
    text: str | None = None
    is_command: bool = False
    is_interrupt: bool = False
    is_rejection: bool = False


def _parse_user(raw: dict[str, Any]) -> _UserParts:
    """Что в user-записи написал сам пользователь; всё подмешанное харнессом — пусто."""
    if raw.get("isMeta") or raw.get("isSidechain") or raw.get("isCompactSummary"):
        return _UserParts()
    origin = raw.get("origin")
    if isinstance(origin, dict) and origin.get("kind") not in (None, "human"):
        return _UserParts()
    content = (raw.get("message") or {}).get("content")
    if isinstance(content, str):
        return _from_texts([content])
    if not isinstance(content, list):
        return _UserParts()
    blocks = [b for b in content if isinstance(b, dict)]
    results = [b for b in blocks if b.get("type") == "tool_result"]
    texts = [str(b.get("text", "")) for b in blocks if b.get("type") == "text"]
    if not results:
        return _from_texts(texts)
    reasons = [r for r in (_rejection_reason(b) for b in results) if r is not None]
    if not reasons:
        return _UserParts()
    parts = _from_texts(texts)
    reason = "\n".join(filter(None, [*reasons, parts.text])).strip()
    return _UserParts(text=reason or None, is_rejection=True)


def _from_texts(texts: list[str]) -> _UserParts:
    """Блоки чистятся по одному: напоминание первым блоком не губит ответ пользователя."""
    cleaned = [clean_reply(t) for t in texts]
    text = "\n".join(c.text for c in cleaned if c.text)
    return _UserParts(
        text=text or None,
        is_command=any(c.is_command for c in cleaned),
        is_interrupt=any(c.is_interrupt for c in cleaned),
    )


def _rejection_reason(block: dict[str, Any]) -> str | None:
    """None — не отказ пользователя; "" — отказ без причины. Отказы automode и правил не в счёт."""
    body = block.get("content")
    if block.get("type") != "tool_result" or not isinstance(body, str):
        return None
    if not body.startswith(_REJECTION_PREFIX):
        return None
    _, sep, reason = body.partition(_REJECTION_REASON)
    return reason.strip() if sep else ""


def _agent_text(raw: dict[str, Any]) -> str | None:
    content = (raw.get("message") or {}).get("content")
    if isinstance(content, str):
        return content.strip() or None
    if not isinstance(content, list):
        return None
    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    text = "\n".join(parts).strip()
    return text or None
