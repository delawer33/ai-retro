import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from ai_retro.classify.service import classify_episode
from ai_retro.cli import app
from ai_retro.db.models import ClassificationRow, EpisodeRow
from ai_retro.errors import ModelCallError
from ai_retro.scan.service import parse_since, scan
from tests.conftest import NO_SKILLS, user_line


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return json.dumps({"category": "запрос", "gist": "делает X"})


def _count(engine: Engine, model: type) -> int:
    with Session(engine) as s:
        return s.scalar(select(func.count()).select_from(model)) or 0


def test_classify_episode_with_fake_model() -> None:
    result = classify_episode(FakeModel(), "привет")
    assert (result.category, result.gist) == ("запрос", "делает X")


def test_classify_rejects_bad_answer() -> None:
    with pytest.raises(ModelCallError):
        classify_episode(lambda _: json.dumps({"category": "???", "gist": "g"}), "x")


def test_scan_is_idempotent(engine: Engine, logs_dir: Path) -> None:
    model = FakeModel()
    first = scan(
        engine,
        model,
        logs_dir=logs_dir,
        since=None,
        threads=2,
        classifier_version="v",
        skills_dir=NO_SKILLS,
    )
    assert (first.found, first.classified) == (2, 2)
    second = scan(
        engine,
        model,
        logs_dir=logs_dir,
        since=None,
        threads=2,
        classifier_version="v",
        skills_dir=NO_SKILLS,
    )
    assert (second.found, second.classified) == (2, 0)
    assert len(model.calls) == 2
    assert _count(engine, EpisodeRow) == 2
    assert _count(engine, ClassificationRow) == 2


def test_scan_since_filters(engine: Engine, logs_dir: Path) -> None:
    since = datetime(2026, 9, 25, 11, 0, 30, tzinfo=UTC)
    result = scan(
        engine,
        FakeModel(),
        logs_dir=logs_dir,
        since=since,
        threads=1,
        classifier_version="v",
        skills_dir=NO_SKILLS,
    )
    assert result.found == 1


def test_parse_since() -> None:
    now = datetime(2026, 10, 4, tzinfo=UTC)
    assert parse_since("1d", now=now) == datetime(2026, 10, 3, tzinfo=UTC)


def test_cli_missing_logs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI_RETRO_LOGS_DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(tmp_path / "db" / "x.db"))
    monkeypatch.setattr("ai_retro.llm.shutil.which", lambda _: "/usr/bin/claude")
    result = CliRunner().invoke(app, ["scan"])
    assert result.exit_code != 0
    assert "Каталог логов не найден" in result.output


def test_cli_missing_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_retro.llm.shutil.which", lambda _: None)
    result = CliRunner().invoke(app, ["scan"])
    assert result.exit_code != 0
    assert "claude" in result.output


def test_scan_rewind_stores_signal_and_stays_idempotent(engine: Engine, tmp_path: Path) -> None:
    project = tmp_path / "logs" / "-p"
    project.mkdir(parents=True)
    asst = json.dumps(
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "timestamp": "2026-09-25T11:00:30.000Z",
            "message": {"content": [{"type": "text", "text": "ок"}]},
        }
    )
    head = [user_line("u1", "корень", "2026-09-25T11:00:00.000Z"), asst]
    (project / "old.jsonl").write_text(
        "\n".join([*head, user_line("u2", "старая", "2026-09-25T11:01:00.000Z", parent="a1")]),
        encoding="utf-8",
    )
    (project / "new.jsonl").write_text(
        "\n".join([*head, user_line("n2", "новая", "2026-09-25T11:05:00.000Z", parent="a1")]),
        encoding="utf-8",
    )
    model = FakeModel()
    first = scan(
        engine,
        model,
        logs_dir=tmp_path / "logs",
        since=None,
        threads=1,
        classifier_version="v",
        skills_dir=NO_SKILLS,
    )
    second = scan(
        engine,
        model,
        logs_dir=tmp_path / "logs",
        since=None,
        threads=1,
        classifier_version="v",
        skills_dir=NO_SKILLS,
    )
    assert (first.found, first.classified, second.found, second.classified) == (3, 3, 3, 0)
    with Session(engine) as s:
        row = s.get(EpisodeRow, "n2")
        assert row is not None
        assert row.signals == ["rewind"]
        assert "старая" in (row.context or "")
        assert _count(engine, EpisodeRow) == 3
