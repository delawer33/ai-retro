import json
import subprocess
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from ai_retro.classify.service import (
    GUIDE_PATH,
    LOOP_MARKER,
    SYSTEM_PROMPT,
    EpisodeInput,
    build_prompt,
    classifier_version,
    classify_many,
    prompt_sha,
)
from ai_retro.cli import app
from ai_retro.config import Settings
from ai_retro.db.models import ClassificationRow
from ai_retro.errors import ModelCallError
from ai_retro.scan.service import scan

from .conftest import user_line

OK = json.dumps({"category": "запрос", "gist": ""})


def test_guide_is_system_prompt_with_marker_and_edge_cases() -> None:
    assert SYSTEM_PROMPT == GUIDE_PATH.read_text(encoding="utf-8")
    assert LOOP_MARKER in SYSTEM_PROMPT
    assert "у нас pathlib" in SYSTEM_PROMPT
    assert "поправка > инструкция > запрос > нейтрально" in SYSTEM_PROMPT


def test_guide_has_qa_edge_cases() -> None:
    assert "Добавь ещё" in SYSTEM_PROMPT
    assert "почему тут 50–60%" in SYSTEM_PROMPT
    assert "в этом разговоре" in SYSTEM_PROMPT
    assert "по форме реплики, не по теме" in SYSTEM_PROMPT
    assert "не повторяет" in SYSTEM_PROMPT


def test_prompt_sha_matches_git_hash_object() -> None:
    expected = subprocess.run(
        ["git", "hash-object", str(GUIDE_PATH)], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert prompt_sha() == expected
    assert classifier_version("haiku") == f"{expected[:12]}+haiku"


def test_build_prompt_has_context_and_signals() -> None:
    prompt = build_prompt(
        EpisodeInput(text="нет, не так", signals=["прерывание"], agent_reply="сделал X"),
        context_chars=100,
    )
    assert "Признаки: прерывание" in prompt
    assert "сделал X" in prompt
    assert prompt.endswith("Реплика пользователя:\nнет, не так")
    assert "брошенной" not in prompt


def test_build_prompt_includes_abandoned_branch_on_rewind() -> None:
    episode = EpisodeInput(text="новая", signals=["rewind"], abandoned_branch="старая ветка")
    assert "Начало брошенной ветки:\nстарая ветка" in build_prompt(episode, context_chars=100)


def test_build_prompt_truncates_by_length() -> None:
    reply = "а" * 50 + "конец"
    prompt = build_prompt(EpisodeInput(text="t", agent_reply=reply), context_chars=10)
    assert "…" + reply[-10:] in prompt
    assert "а" * 11 not in prompt


def test_classify_many_keeps_processed_on_error() -> None:
    def call(prompt: str) -> str:
        return "не json" if prompt == "bad" else OK

    outcome = classify_many(call, [("a", "ok"), ("b", "bad")], threads=1)
    assert set(outcome.results) == {"a"}
    assert isinstance(outcome.error, ModelCallError)


def test_scan_saves_processed_then_raises(engine: Engine, logs_dir: Path) -> None:
    def call(prompt: str) -> str:
        return "мусор" if "я же просил" in prompt else OK

    with pytest.raises(ModelCallError):
        scan(engine, call, logs_dir=logs_dir, since=None, threads=1, classifier_version="v1")
    with Session(engine) as s:
        rows = s.scalars(select(ClassificationRow)).all()
    assert [(r.episode_uuid, r.classifier_version) for r in rows] == [("u1", "v1")]


def test_scan_gives_agent_reply_as_context(engine: Engine, logs_dir: Path) -> None:
    prompts: list[str] = []

    def call(prompt: str) -> str:
        prompts.append(prompt)
        return OK

    scan(engine, call, logs_dir=logs_dir, since=None, threads=1, classifier_version="v")
    second = next(p for p in prompts if "я же просил" in p)
    assert "Последний ответ агента перед репликой:\nok" in second


def test_scan_announces_count_before_first_call(engine: Engine, logs_dir: Path) -> None:
    events: list[str] = []

    def call(_: str) -> str:
        events.append("call")
        return OK

    scan(
        engine,
        call,
        logs_dir=logs_dir,
        since=None,
        threads=1,
        classifier_version="v",
        announce=lambda n: events.append(f"announce {n}"),
    )
    assert events[0] == "announce 2"
    assert events.count("call") == 2


def test_scan_skips_conversation_with_loop_marker(engine: Engine, logs_dir: Path) -> None:
    other = logs_dir / "-work-other"
    other.mkdir()
    lines = [user_line("x1", f"{LOOP_MARKER} разбери", "2026-09-25T12:00:00.000Z")]
    (other / "s2.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    calls: list[str] = []

    def call(prompt: str) -> str:
        calls.append(prompt)
        return OK

    result = scan(engine, call, logs_dir=logs_dir, since=None, threads=1, classifier_version="v")
    assert result.found == 2
    assert not any(LOOP_MARKER in p for p in calls)


def test_threads_default_in_range() -> None:
    assert 2 <= Settings().threads <= 4


def test_cli_prints_count_before_model_and_version_stored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, logs_dir: Path
) -> None:
    monkeypatch.setenv("AI_RETRO_LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(tmp_path / "db" / "x.db"))
    monkeypatch.setenv("AI_RETRO_MODEL", "haiku")
    monkeypatch.setattr("ai_retro.cli.make_claude_call", lambda **_: lambda _p: OK)
    result = CliRunner().invoke(app, ["scan"])
    assert result.exit_code == 0, result.output
    assert "Вызовов к модели (эпизоды и области): 2" in result.output
    from ai_retro.db.session import make_engine

    with Session(make_engine(tmp_path / "db" / "x.db")) as s:
        versions = set(s.scalars(select(ClassificationRow.classifier_version)))
    assert versions == {classifier_version("haiku")}


def test_cli_model_error_exits_nonzero_with_one_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, logs_dir: Path
) -> None:
    monkeypatch.setenv("AI_RETRO_LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(tmp_path / "db" / "x.db"))
    monkeypatch.setattr("ai_retro.cli.make_claude_call", lambda **_: lambda _p: "мусор")
    result = CliRunner().invoke(app, ["scan"])
    assert result.exit_code == 1
    assert "Ошибка:" in result.output
