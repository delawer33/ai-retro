import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from ai_retro.cli import app
from ai_retro.db.models import ClassificationRow, EpisodeRow, PatternRow, ProposalRow
from ai_retro.errors import ModelCallError, NoClassificationsError, ReportWriteError
from ai_retro.report.proposals import choose_target
from ai_retro.report.schemas import (
    TARGET_GLOBAL,
    TARGET_PROJECT,
    TARGET_SKILL,
    GistItem,
    Pattern,
)
from ai_retro.report.service import (
    build_report,
    check_report_path,
    load_report_data,
    render_report,
    run_report,
    write_report,
)
from ai_retro.report.text import PASTE_MARKER, mask_pastes

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


def proposal_call(*, kind: str = "edit", section: str = "Правила", text: str = "Не пиши X."):  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def call(prompt: str) -> str:
        calls.append(prompt)
        data = {"kind": kind, "section": section, "text": text, "rationale": "Повторяется."}
        return json.dumps(data)

    call.calls = calls  # type: ignore[attr-defined]
    return call


def test_group_from_one_conversation_is_not_a_pattern(engine: Engine) -> None:
    seed(engine, [(f"e{i}", "c1", "поправка", "не пиши X", "t") for i in range(4)])
    data = build_report(
        engine,
        groups_call(("X", [0, 1, 2, 3])),
        proposal_call(),
        since=None,
        now=NOW,
        classifier_version=VERSION,
    )
    assert data.patterns == ()


def test_group_of_two_episodes_is_not_a_pattern(engine: Engine) -> None:
    seed(engine, [(f"e{i}", f"c{i}", "поправка", "не пиши X", "t") for i in range(2)])
    data = build_report(
        engine,
        groups_call(("X", [0, 1])),
        proposal_call(),
        since=None,
        now=NOW,
        classifier_version=VERSION,
    )
    assert data.patterns == ()
    with Session(engine) as s:
        assert s.scalars(select(PatternRow)).all() == []


def test_pattern_saved_and_only_correction_and_instruction_grouped(engine: Engine) -> None:
    rows = [(f"e{i}", f"c{i % 2}", "поправка", "не пиши X", f"text {i}") for i in range(3)]
    rows.append(("q", "c9", "запрос", "сделай Y", "сделай Y"))
    seed(engine, rows)
    call = groups_call(("Не писать X", [0, 1, 2, 7, 0]))
    data = build_report(
        engine, call, proposal_call(), since=None, now=NOW, classifier_version=VERSION
    )
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
        proposal_call(),
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
        build_report(
            engine, groups_call(), proposal_call(), since=None, now=NOW, classifier_version=VERSION
        )


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
            run_report(
                engine,
                call,
                proposal_call(),
                output=bad,
                since=None,
                now=NOW,
                classifier_version=VERSION,
            )
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
        proposal_call(),
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


def item(uuid: str, conv: str, project: str = "/work/proj", skill: str | None = None) -> GistItem:
    return GistItem(
        uuid=uuid,
        conversation_id=conv,
        project=project,
        domain="работа",
        category="поправка",
        gist="не пиши X",
        text="не пиши X",
        timestamp=NOW,
        skill=skill,
    )


def pattern_of(*items: GistItem) -> Pattern:
    return Pattern(title="Не писать X", episodes=items)


SCRATCH = "/home/u/.config/Claude/scratch-workspaces/abc"


def test_target_one_project_is_that_projects_claude_md() -> None:
    target = choose_target(pattern_of(item("a", "c1"), item("b", "c2"), item("c", "c3")))
    assert (target.kind, target.file) == (TARGET_PROJECT, "/work/proj/CLAUDE.md")


def test_target_different_projects_is_global() -> None:
    target = choose_target(
        pattern_of(item("a", "c1", "/work/a"), item("b", "c2", "/work/b"), item("c", "c3"))
    )
    assert (target.kind, target.file) == (TARGET_GLOBAL, "~/.claude/CLAUDE.md")


def test_target_scratch_is_global_even_if_one_path() -> None:
    items = [item(f"e{i}", f"c{i}", SCRATCH) for i in range(3)]
    assert choose_target(pattern_of(*items)).kind == TARGET_GLOBAL


def test_target_skill_when_most_episodes_follow_one_skill() -> None:
    items = [item("a", "c1", skill="feat"), item("b", "c2", skill="feat"), item("c", "c3")]
    target = choose_target(pattern_of(*items))
    assert (target.kind, target.file) == (TARGET_SKILL, "~/.claude/skills/feat/SKILL.md")


def test_target_skill_needs_a_majority() -> None:
    items = [item("a", "c1", skill="feat"), item("b", "c2"), item("c", "c3"), item("d", "c4")]
    assert choose_target(pattern_of(*items)).kind == TARGET_PROJECT


def seed_skill_conversation(engine: Engine) -> None:
    with Session(engine) as s:
        s.add(
            EpisodeRow(
                uuid="call",
                conversation_id="c0",
                project="/work/proj",
                timestamp=datetime(2026, 9, 1, tzinfo=UTC),
                text="/feat добавь кнопку",
                signals=["вызов скилла"],
                skill="feat",
            )
        )
        s.commit()


def test_episode_after_skill_call_is_linked_to_skill(engine: Engine) -> None:
    seed_skill_conversation(engine)
    rows = [(f"e{i}", "c0" if i < 2 else "c1", "поправка", "не пиши X", f"t{i}") for i in range(3)]
    seed(engine, rows)
    items, _ = load_report_data(engine, since=None, classifier_version=VERSION)
    assert [i.skill for i in items] == ["feat", "feat", None]


def test_proposal_saved_as_proposed_and_matches_report(engine: Engine, tmp_path: Path) -> None:
    rows = [(f"e{i}", f"c{i % 2}", "поправка", "не пиши X", f"t{i}") for i in range(3)]
    seed(engine, rows)
    out = tmp_path / "r.md"
    run_report(
        engine,
        groups_call(("Не писать X", [0, 1, 2])),
        proposal_call(section="Стиль", text="Не пиши X.\nНикогда."),
        output=out,
        since=None,
        now=NOW,
        classifier_version=VERSION,
    )
    with Session(engine) as s:
        (row,) = s.scalars(select(ProposalRow)).all()
    assert (row.status, row.target_file, row.section) == (
        "proposed",
        "/work/proj/CLAUDE.md",
        "Стиль",
    )
    md = out.read_text(encoding="utf-8")
    assert "предложено" in md
    assert "`/work/proj/CLAUDE.md`, раздел «Стиль»" in md
    assert "> Не пиши X.\n    > Никогда." in md
    assert "обоснование: Повторяется." in md
    assert row.text == "Не пиши X.\nНикогда."


def test_new_skill_proposal_has_no_target_file(engine: Engine, tmp_path: Path) -> None:
    rows = [(f"e{i}", f"c{i % 2}", "поправка", "не пиши X", f"t{i}") for i in range(3)]
    seed(engine, rows)
    out = tmp_path / "r.md"
    run_report(
        engine,
        groups_call(("Не писать X", [0, 1, 2])),
        proposal_call(kind="new_skill", section="", text="Скилл: делает Y."),
        output=out,
        since=None,
        now=NOW,
        classifier_version=VERSION,
    )
    with Session(engine) as s:
        (row,) = s.scalars(select(ProposalRow)).all()
    assert row.target_file == ""
    assert "новый скилл" in out.read_text(encoding="utf-8")


def test_proposal_prompt_names_target_file(engine: Engine) -> None:
    rows = [(f"e{i}", f"c{i % 2}", "поправка", "не пиши X", f"t{i}") for i in range(3)]
    seed(engine, rows)
    pc = proposal_call()
    build_report(
        engine,
        groups_call(("Не писать X", [0, 1, 2])),
        pc,
        since=None,
        now=NOW,
        classifier_version=VERSION,
    )
    assert len(pc.calls) == 1
    assert "/work/proj/CLAUDE.md" in pc.calls[0]


def test_second_report_reuses_stored_proposal(engine: Engine) -> None:
    rows = [(f"e{i}", f"c{i % 2}", "поправка", "не пиши X", f"t{i}") for i in range(3)]
    seed(engine, rows)
    for text in ("Первый текст.", "Второй текст."):
        data = build_report(
            engine,
            groups_call(("Не писать X", [0, 1, 2])),
            proposal_call(text=text),
            since=None,
            now=NOW,
            classifier_version=VERSION,
        )
    (pattern,) = data.patterns
    assert pattern.proposal is not None
    assert pattern.proposal.text == "Первый текст."
    with Session(engine) as s:
        assert len(s.scalars(select(ProposalRow)).all()) == 1
        assert len(s.scalars(select(PatternRow)).all()) == 1


def test_bad_proposal_answer_is_model_call_error(engine: Engine) -> None:
    rows = [(f"e{i}", f"c{i % 2}", "поправка", "не пиши X", f"t{i}") for i in range(3)]
    seed(engine, rows)
    with pytest.raises(ModelCallError):
        build_report(
            engine,
            groups_call(("Не писать X", [0, 1, 2])),
            lambda _: "не json",
            since=None,
            now=NOW,
            classifier_version=VERSION,
        )
    with Session(engine) as s:
        assert s.scalars(select(PatternRow)).all() == []


def test_proposal_never_touches_target_files(engine: Engine, tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    claude_md = project / "CLAUDE.md"
    claude_md.write_text("исходный", encoding="utf-8")
    with Session(engine) as s:
        for i in range(3):
            s.add(
                EpisodeRow(
                    uuid=f"e{i}",
                    conversation_id=f"c{i % 2}",
                    project=str(project),
                    timestamp=NOW,
                    text="t",
                    signals=[],
                )
            )
            s.add(
                ClassificationRow(
                    episode_uuid=f"e{i}",
                    category="поправка",
                    gist="не пиши X",
                    classifier_version=VERSION,
                )
            )
        s.commit()
    build_report(
        engine,
        groups_call(("Не писать X", [0, 1, 2])),
        proposal_call(),
        since=None,
        now=NOW,
        classifier_version=VERSION,
    )
    assert claude_md.read_text(encoding="utf-8") == "исходный"
