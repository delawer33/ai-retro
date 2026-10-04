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
    first = scan(engine, model, logs_dir=logs_dir, since=None, threads=2)
    assert (first.found, first.classified) == (2, 2)
    second = scan(engine, model, logs_dir=logs_dir, since=None, threads=2)
    assert (second.found, second.classified) == (2, 0)
    assert len(model.calls) == 2
    assert _count(engine, EpisodeRow) == 2
    assert _count(engine, ClassificationRow) == 2


def test_scan_since_filters(engine: Engine, logs_dir: Path) -> None:
    since = datetime(2026, 9, 25, 11, 0, 30, tzinfo=UTC)
    result = scan(engine, FakeModel(), logs_dir=logs_dir, since=since, threads=1)
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
