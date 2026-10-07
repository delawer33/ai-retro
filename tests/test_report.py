import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from ai_retro.cli import app
from ai_retro.db.models import ClassificationRow, EpisodeRow, PatternRow
from ai_retro.errors import NoClassificationsError, ReportWriteError
from ai_retro.report.service import (
    PASTE_MARKER,
    build_report,
    check_report_path,
    mask_pastes,
    render_report,
    run_report,
    write_report,
)

NOW = datetime(2026, 10, 4, tzinfo=UTC)
VERSION = "v1"


def seed(engine: Engine, rows: list[tuple[str, str, str, str, str]]) -> None:
    """rows: (uuid, conversation, category, gist, text)."""
    with Session(engine) as s:
        for i, (uuid, conv, category, gist, text) in enumerate(rows):
            s.add(
                EpisodeRow(
                    uuid=uuid,
                    conversation_id=conv,
                    project="/work/proj",
                    domain="работа" if i % 2 == 0 else None,
                    timestamp=datetime(2026, 10, 1, tzinfo=UTC),
                    text=text,
                    signals=[],
                )
            )
            s.add(
                ClassificationRow(
                    episode_uuid=uuid,
                    category=category,
                    gist=gist,
                    classifier_version=VERSION,
                )
            )
        s.commit()


def groups_call(*groups: tuple[str, list[int]]):  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def call(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps({"groups": [{"title": t, "ids": ids} for t, ids in groups]})

    call.calls = calls  # type: ignore[attr-defined]
    return call


def test_group_from_one_conversation_is_not_a_pattern(engine: Engine) -> None:
    seed(engine, [(f"e{i}", "c1", "поправка", "не пиши X", "t") for i in range(4)])
    data = build_report(
        engine, groups_call(("X", [0, 1, 2, 3])), since=None, now=NOW, classifier_version=VERSION
    )
    assert data.patterns == ()


def test_group_of_two_episodes_is_not_a_pattern(engine: Engine) -> None:
    seed(engine, [(f"e{i}", f"c{i}", "поправка", "не пиши X", "t") for i in range(2)])
    data = build_report(
        engine, groups_call(("X", [0, 1])), since=None, now=NOW, classifier_version=VERSION
    )
    assert data.patterns == ()
    with Session(engine) as s:
        assert s.scalars(select(PatternRow)).all() == []


def test_pattern_saved_and_only_correction_and_instruction_grouped(engine: Engine) -> None:
    rows = [(f"e{i}", f"c{i % 2}", "поправка", "не пиши X", f"text {i}") for i in range(3)]
    rows.append(("q", "c9", "запрос", "сделай Y", "сделай Y"))
    seed(engine, rows)
    call = groups_call(("Не писать X", [0, 1, 2, 7, 0]))
    data = build_report(engine, call, since=None, now=NOW, classifier_version=VERSION)
    assert len(call.calls) == 1
    assert "сделай Y" not in call.calls[0]
    (pattern,) = data.patterns
    assert (len(pattern.episodes), len(pattern.conversations)) == (3, 2)
    with Session(engine) as s:
        (row,) = s.scalars(select(PatternRow)).all()
        assert row.gist == "Не писать X"
        assert sorted(row.episode_uuids) == ["e0", "e1", "e2"]


def test_render_has_all_sections(engine: Engine) -> None:
    rows = [
        (f"e{i}", f"c{i % 2}", "инструкция", "всегда делай X", f"реплика {i}") for i in range(3)
    ]
    seed(engine, rows)
    data = build_report(
        engine,
        groups_call(("Всегда делать X", [0, 1, 2])),
        since=None,
        now=NOW,
        classifier_version=VERSION,
    )
    md = render_report(data)
    assert "## Категории по областям" in md
    assert "| работа |" in md
    assert "### Всегда делать X" in md
    assert "эпизодов: 3, бесед: 2" in md
    assert "области: без области, работа" in md
    assert "проекты: /work/proj" in md
    assert "реплика 0" in md
    assert "беседа `c0`" in md
    assert "## Топ-10" in md
    assert "перепроверить:" in md


def test_pastes_shown_as_marker() -> None:
    text = (
        "смотри\n```\nsecret log\n```\nи [Pasted text #1 +30 lines] и "
        "<pasted_content>long log</pasted_content> ещё"
    )
    masked = mask_pastes(text)
    assert "secret log" not in masked
    assert "Pasted text" not in masked
    assert "long log" not in masked
    assert masked.count(PASTE_MARKER[:-1]) == 3
    assert "симв.]" in masked


def test_no_classifications_raises(engine: Engine) -> None:
    with pytest.raises(NoClassificationsError, match="scan"):
        build_report(engine, groups_call(), since=None, now=NOW, classifier_version=VERSION)


def test_report_never_writes_claude_md(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="CLAUDE.md"):
        write_report(tmp_path / "CLAUDE.md", "x")
    assert not (tmp_path / "CLAUDE.md").exists()


def test_protected_path_case_insensitive(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="claude.md"):
        check_report_path(tmp_path / "claude.md")


def test_cli_rejects_protected_output_before_model_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(tmp_path / "db" / "x.db"))
    result = CliRunner().invoke(app, ["report", "--output", str(tmp_path / "CLAUDE.md")])
    assert result.exit_code == 1
    assert not (tmp_path / "db" / "x.db").exists()


def test_cli_report_without_classifications(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(tmp_path / "db" / "x.db"))
    monkeypatch.setattr("ai_retro.llm.shutil.which", lambda _: "/usr/bin/claude")
    result = CliRunner().invoke(app, ["report", "--output", str(tmp_path / "r.md")])
    assert result.exit_code != 0
    assert "scan" in result.output


def _bad_paths(tmp_path: Path) -> list[Path]:
    (tmp_path / "adir").mkdir()
    paths = [tmp_path / "missing" / "r.md", tmp_path / "adir"]
    if os.geteuid() != 0:
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(0o500)
        paths.append(ro / "r.md")
    return paths


def test_write_report_translates_os_errors(tmp_path: Path) -> None:
    for bad in _bad_paths(tmp_path):
        with pytest.raises(ReportWriteError, match="Укажи другой --output"):
            write_report(bad, "x")


def test_write_failure_after_check_is_domain_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: Path, *args: object, **kwargs: object) -> int:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(ReportWriteError, match="Permission denied"):
        write_report(tmp_path / "r.md", "x")


def test_bad_output_path_costs_no_model_call_and_saves_no_patterns(
    engine: Engine, tmp_path: Path
) -> None:
    rows = [(f"e{i}", f"c{i % 2}", "поправка", "не пиши X", f"t{i}") for i in range(3)]
    seed(engine, rows)
    for bad in _bad_paths(tmp_path):
        call = groups_call(("Не писать X", [0, 1, 2]))
        with pytest.raises(ReportWriteError):
            run_report(engine, call, output=bad, since=None, now=NOW, classifier_version=VERSION)
        assert call.calls == []
        with Session(engine) as s:
            assert s.scalars(select(PatternRow)).all() == []


def test_run_report_writes_file(engine: Engine, tmp_path: Path) -> None:
    rows = [(f"e{i}", f"c{i % 2}", "поправка", "не пиши X", f"t{i}") for i in range(3)]
    seed(engine, rows)
    out = tmp_path / "r.md"
    run_report(
        engine,
        groups_call(("Не писать X", [0, 1, 2])),
        output=out,
        since=None,
        now=NOW,
        classifier_version=VERSION,
    )
    assert "### Не писать X" in out.read_text(encoding="utf-8")


def test_cli_report_bad_output_never_calls_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(tmp_path / "db" / "x.db"))

    def forbidden(**_: object) -> None:
        raise AssertionError("модель не должна вызываться")

    monkeypatch.setattr("ai_retro.cli.make_claude_call", forbidden)
    result = CliRunner().invoke(app, ["report", "--output", str(tmp_path / "no" / "r.md")])
    assert result.exit_code == 1
    assert "Ошибка:" in result.output
    assert "OSError" not in result.output
