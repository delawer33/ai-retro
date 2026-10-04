"""Стык «файлы → записи»."""

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_retro.parsing.schemas import Record


def find_log_files(logs_dir: Path) -> list[Path]:
    return sorted(logs_dir.glob("*/*.jsonl"))


def files_to_records(files: Iterable[Path]) -> list[Record]:
    records: list[Record] = []
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                record = _parse_line(line, path)
                if record is not None:
                    records.append(record)
    return records


def _parse_line(line: str, source: Path) -> Record | None:
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
    return Record(
        uuid=uuid,
        parent_uuid=raw.get("parentUuid"),
        session_id=raw.get("sessionId", source.stem),
        project=raw.get("cwd", source.parent.name),
        timestamp=datetime.fromisoformat(timestamp),
        kind=kind,
        text=_user_text(raw) if kind == "user" else None,
        source=source,
        assistant_text=_assistant_text(raw) if kind == "assistant" else None,
    )


def _user_text(raw: dict[str, Any]) -> str | None:
    """Текст, написанный самим пользователем; всё подмешанное харнессом — None."""
    if raw.get("isMeta") or raw.get("isSidechain"):
        return None
    origin = raw.get("origin")
    if isinstance(origin, dict) and origin.get("kind") not in (None, "human"):
        return None
    content = (raw.get("message") or {}).get("content")
    if isinstance(content, list):
        parts = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        content = "\n".join(parts) if len(parts) == len(content) else None
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def _assistant_text(raw: dict[str, Any]) -> str | None:
    content = (raw.get("message") or {}).get("content")
    if isinstance(content, str):
        return content or None
    if not isinstance(content, list):
        return None
    parts = [
        b["text"]
        for b in content
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
    ]
    return "\n".join(parts) or None
