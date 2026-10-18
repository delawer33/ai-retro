import json
from pathlib import Path

import pytest
from sqlalchemy import Engine

from ai_retro.db.session import make_engine, migrate

NO_SKILLS = Path("/nonexistent/skills")  # ни одного скилла: слэш-команды — обычные реплики


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    engine = make_engine(tmp_path / "test.db")
    migrate(engine)
    return engine


def user_line(uuid: str, text: str, ts: str, *, parent: str | None = None, **extra: object) -> str:
    return json.dumps(
        {
            "type": "user",
            "uuid": uuid,
            "parentUuid": parent,
            "timestamp": ts,
            "sessionId": "s1",
            "cwd": "/work/proj",
            "message": {"role": "user", "content": text},
            **extra,
        }
    )


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    project = tmp_path / "logs" / "-work-proj"
    project.mkdir(parents=True)
    lines = [
        json.dumps({"type": "queue-operation", "operation": "enqueue"}),
        user_line("u1", "сделай X", "2026-09-25T11:00:00.000Z"),
        json.dumps(
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "u1",
                "timestamp": "2026-09-25T11:00:05.000Z",
                "sessionId": "s1",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            }
        ),
        user_line("u2", "не так, я же просил Y", "2026-09-25T11:01:00.000Z", parent="a1"),
        json.dumps(
            {
                "type": "user",
                "uuid": "t1",
                "parentUuid": "u2",
                "timestamp": "2026-09-25T11:01:05.000Z",
                "sessionId": "s1",
                "message": {"role": "user", "content": [{"type": "tool_result", "content": "r"}]},
            }
        ),
        user_line("m1", "harness note", "2026-09-25T11:02:00.000Z", isMeta=True),
        "not json",
    ]
    (project / "s1.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path / "logs"


def make_skill(root: Path, name: str) -> Path:
    """Скилл в корне `root`: `<root>/<name>/SKILL.md`; возвращает корень."""
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    return root
