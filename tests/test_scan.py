import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from ai_retro.classify.service import classify_episode
from ai_retro.cli import app
from ai_retro.db.models import (
    ClassificationRow,
    EpisodeRow,
    GoldLabelRow,
    GoldReturnedRow,
    GoldSampleRow,
)
from ai_retro.errors import ModelCallError
from ai_retro.scan.service import Review, ScanResult, parse_since, scan
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


def _three_episodes(tmp_path: Path) -> Path:
    project = tmp_path / "logs" / "-p"
    project.mkdir(parents=True)
    for i, text in enumerate(["первая", "вторая", "третья"]):
        (project / f"s{i}.jsonl").write_text(
            user_line(f"u{i}", text, f"2026-09-25T11:0{i}:00.000Z") + "\n", encoding="utf-8"
        )
    return tmp_path / "logs"


def test_scan_skips_schema_miss_and_reports_it(
    engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs = _three_episodes(tmp_path)

    def model(prompt: str) -> str:
        if "Реплика пользователя" in prompt and "вторая" in prompt:
            return "Я не могу вызвать StructuredOutput"
        return json.dumps({"category": "запрос", "gist": ""})

    monkeypatch.setenv("AI_RETRO_LOGS_DIR", str(logs))
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(tmp_path / "db" / "x.db"))
    monkeypatch.setattr("ai_retro.cli.make_claude_call", lambda **_: model)
    result = CliRunner().invoke(app, ["scan"])
    assert result.exit_code == 0, result.output
    assert "Не классифицировано: 1 (ответ не по схеме)" in result.output
    assert "классифицировано: 2" in result.output


def test_scan_retries_schema_miss_once(engine: Engine, tmp_path: Path) -> None:
    logs = _three_episodes(tmp_path)
    attempts: dict[str, int] = {}

    def model(prompt: str) -> str:
        attempts[prompt] = attempts.get(prompt, 0) + 1
        if "вторая" in prompt and attempts[prompt] == 1:
            return "текст"
        return json.dumps({"category": "запрос", "gist": ""})

    result = scan(
        engine,
        model,
        logs_dir=logs,
        since=None,
        threads=1,
        classifier_version="v",
        skills_dir=NO_SKILLS,
    )
    assert (result.classified, result.unclassified) == (3, 0)
    assert sum(attempts.values()) == 4


def test_cli_unrecognized_model_still_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai_retro.errors import UnrecognizedModelError

    def model(_: str) -> str:
        raise UnrecognizedModelError("nope")

    monkeypatch.setenv("AI_RETRO_LOGS_DIR", str(_three_episodes(tmp_path)))
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(tmp_path / "db" / "x.db"))
    monkeypatch.setattr("ai_retro.cli.make_claude_call", lambda **_: model)
    result = CliRunner().invoke(app, ["scan"])
    assert result.exit_code == 1
    assert "не опознана" in result.output


def _scan(engine: Engine, model: FakeModel, logs: Path) -> ScanResult:
    return scan(
        engine,
        model,
        logs_dir=logs,
        since=None,
        threads=1,
        classifier_version="v",
        skills_dir=NO_SKILLS,
    )


def _stale(uuid: str, conversation_id: str = "s1") -> EpisodeRow:
    return EpisodeRow(
        uuid=uuid,
        conversation_id=conversation_id,
        project="/work/proj",
        domain="иное",
        timestamp=datetime(2026, 9, 25, 11, 0, tzinfo=UTC),
        text="<local-command-stdout>x</local-command-stdout>",
        signals=[],
    )


def test_scan_review_removes_junk_with_classification(engine: Engine, logs_dir: Path) -> None:
    with Session(engine) as s:
        s.add(_stale("junk"))
        s.add(
            ClassificationRow(
                episode_uuid="junk", category="запрос", gist="g", classifier_version="v"
            )
        )
        s.commit()
    seen: list[Review] = []
    model = FakeModel()
    result = scan(
        engine,
        model,
        logs_dir=logs_dir,
        since=None,
        threads=1,
        classifier_version="v",
        reviewed=seen.append,
        skills_dir=NO_SKILLS,
    )
    assert (result.review.removed, result.review.removed_gold_labels) == (1, 0)
    assert seen == [result.review]
    with Session(engine) as s:
        assert s.get(EpisodeRow, "junk") is None
        assert not s.scalars(
            select(ClassificationRow).where(ClassificationRow.episode_uuid == "junk")
        ).all()


def test_scan_review_removes_builtin_command_episode(engine: Engine, tmp_path: Path) -> None:
    command = "<command-message>compact</command-message>\n<command-name>/compact</command-name>"
    with Session(engine) as s:
        s.add(_stale("cmd"))
        s.add(
            ClassificationRow(
                episode_uuid="cmd", category="поправка", gist="g", classifier_version="v"
            )
        )
        s.commit()
    project = tmp_path / "logs" / "-p"
    project.mkdir(parents=True)
    line = user_line("cmd", command, "2026-09-25T11:00:00.000Z")
    (project / "s1.jsonl").write_text(line, encoding="utf-8")
    result = _scan(engine, FakeModel(), tmp_path / "logs")
    assert result.review.removed == 1
    with Session(engine) as s:
        assert s.get(EpisodeRow, "cmd") is None
        assert not s.scalars(select(ClassificationRow)).all()


def test_scan_review_removes_artifact_view_context_episode(engine: Engine, logs_dir: Path) -> None:
    stale = _stale("junk")
    stale.text = (
        '<artifact-view-context artifact="a1"> {"context":{"mode":"edit"}} </artifact-view-context>'
    )
    with Session(engine) as s:
        s.add(stale)
        s.commit()
    result = _scan(engine, FakeModel(), logs_dir)
    assert result.review.removed == 1
    with Session(engine) as s:
        assert s.get(EpisodeRow, "junk") is None


def test_scan_review_removes_gold_rows(engine: Engine, logs_dir: Path) -> None:
    with Session(engine) as s:
        s.add(_stale("junk"))
        s.flush()
        s.add(
            GoldLabelRow(
                episode_uuid="junk",
                category="запрос",
                split="dev",
                labeled_at=datetime(2026, 9, 26, tzinfo=UTC),
            )
        )
        s.add(GoldSampleRow(episode_uuid="junk", split="dev"))
        s.add(GoldReturnedRow(episode_uuid="junk"))
        s.commit()
    result = _scan(engine, FakeModel(), logs_dir)
    assert result.review.removed_gold_labels == 1
    for table in (GoldLabelRow, GoldSampleRow, GoldReturnedRow):
        assert _count(engine, table) == 0


def test_scan_review_updates_changed_episode_keeps_classification(
    engine: Engine, logs_dir: Path
) -> None:
    model = FakeModel()
    _scan(engine, model, logs_dir)
    with Session(engine) as s:
        row = s.get(EpisodeRow, "u2")
        assert row is not None
        row.text = "старый текст"
        s.commit()
    result = _scan(engine, model, logs_dir)
    assert (result.review.removed, result.review.updated) == (0, 1)
    assert len(model.calls) == 2
    with Session(engine) as s:
        row = s.get(EpisodeRow, "u2")
        assert row is not None
        assert row.text == "не так, я же просил Y"
    assert _count(engine, ClassificationRow) == 2


def test_scan_review_noop_on_repeat(engine: Engine, logs_dir: Path) -> None:
    model = FakeModel()
    _scan(engine, model, logs_dir)
    result = _scan(engine, model, logs_dir)
    assert (result.review.removed, result.review.updated) == (0, 0)
    assert len(model.calls) == 2


def test_scan_review_leaves_conversations_outside_window(engine: Engine, logs_dir: Path) -> None:
    with Session(engine) as s:
        s.add(_stale("junk"))
        s.commit()
    since = datetime(2026, 10, 1, tzinfo=UTC)
    result = scan(
        engine,
        FakeModel(),
        logs_dir=logs_dir,
        since=since,
        threads=1,
        classifier_version="v",
        skills_dir=NO_SKILLS,
    )
    assert result.review.removed == 0
    assert _count(engine, EpisodeRow) == 1


def test_cli_scan_prints_review(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, logs_dir: Path
) -> None:
    monkeypatch.setenv("AI_RETRO_LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(tmp_path / "db" / "x.db"))
    monkeypatch.setattr("ai_retro.llm.shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr("ai_retro.cli.make_claude_call", lambda **_: FakeModel())
    result = CliRunner().invoke(app, ["scan"])
    assert "Пересмотр: удалено 0 эпизодов, обновлено 0" in result.output
