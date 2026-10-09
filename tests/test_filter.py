"""Полный фильтр реплик и признаки; фикстуры синтетические, по форме реальных логов."""

import json
from pathlib import Path

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from ai_retro.db.models import ClassificationRow, EpisodeRow
from ai_retro.parsing.conversations import records_to_conversations
from ai_retro.parsing.episodes import clean_reply, conversations_to_episodes
from ai_retro.parsing.files import files_to_records, find_log_files
from ai_retro.parsing.schemas import EpisodeDraft
from ai_retro.scan.service import scan
from tests.conftest import NO_SKILLS, make_skill, user_line

REJECT = (
    "The user doesn't want to proceed with this tool use. The tool use was rejected "
    "(eg. if it was a file edit, the new_string was NOT written to the file). "
    "STOP what you are doing and wait for the user to tell you how to proceed."
)
AUTOMODE = "Permission for this action was denied by the Claude Code auto mode classifier."
RULE = "Permission to use Bash with command ls has been denied."


def _ts(n: int) -> str:
    return f"2026-09-25T11:{n:02d}:00.000Z"


def _blocks(uuid: str, n: int, blocks: list[dict[str, object]], **extra: object) -> str:
    return json.dumps(
        {
            "type": "user",
            "uuid": uuid,
            "timestamp": _ts(n),
            "sessionId": "s1",
            "cwd": "/work/proj",
            "message": {"role": "user", "content": blocks},
            **extra,
        }
    )


def _tool_result(uuid: str, n: int, content: str) -> str:
    block = {"type": "tool_result", "content": content, "is_error": True, "tool_use_id": "t"}
    return _blocks(uuid, n, [block])


def _write(
    root: Path, lines: list[str], *, name: str = "-work-proj", file: str = "s1.jsonl"
) -> Path:
    project = root / "logs" / name
    project.mkdir(parents=True, exist_ok=True)
    (project / file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root / "logs"


def _episodes(logs: Path, skills_dir: Path = NO_SKILLS) -> list[EpisodeDraft]:
    records = files_to_records(find_log_files(logs))
    return conversations_to_episodes(records_to_conversations(records), skills_dir=skills_dir)


def test_only_user_written_text_is_episode(tmp_path: Path) -> None:
    command = (
        "<command-message>plain</command-message>\n<command-name>/plain</command-name>\n"
        "<command-args>перепиши текст</command-args>"
    )
    lines = [
        user_line("u1", "обычная реплика", _ts(0)),
        _tool_result("r1", 1, "some output"),
        user_line("m1", "тело скилла", _ts(2), isMeta=True),
        user_line("n1", "<task-notification>\n<task-id>x</task-id>\n</task-notification>", _ts(3)),
        user_line("l1", "<local-command-stdout>Set model</local-command-stdout>", _ts(4)),
        user_line("c1", "This session is being continued...", _ts(5), isCompactSummary=True),
        user_line("s2", "<system-reminder>\nnote\n</system-reminder>", _ts(6)),
        user_line("u2", command, _ts(7)),
    ]
    skills = make_skill(tmp_path / "skills", "plain")
    episodes = _episodes(_write(tmp_path, lines), skills)
    assert [(e.uuid, e.text) for e in episodes] == [
        ("u1", "обычная реплика"),
        ("u2", "/plain перепиши текст"),
    ]
    assert [(e.category, e.skill) for e in episodes] == [(None, None), (None, "plain")]
    assert episodes[1].signals == ("вызов скилла",)


def test_reminder_block_before_user_text_keeps_reply(tmp_path: Path) -> None:
    blocks: list[dict[str, object]] = [
        {"type": "text", "text": "<system-reminder>служебное</system-reminder>"},
        {"type": "text", "text": "поправь тест"},
    ]
    episodes = _episodes(_write(tmp_path, [_blocks("u1", 0, blocks)]))
    assert [e.text for e in episodes] == ["поправь тест"]


_ARTIFACT_BLOCK = (
    '<artifact-view-context artifact="a1"> {"context":{"mode":"edit"}} </artifact-view-context>'
)


def test_artifact_view_context_block_is_stripped_from_reply(tmp_path: Path) -> None:
    lines = [
        user_line("u1", _ARTIFACT_BLOCK + "\nпоправь заголовок", _ts(0)),
        user_line("u2", _ARTIFACT_BLOCK, _ts(1)),
    ]
    episodes = _episodes(_write(tmp_path, lines))
    assert [(e.uuid, e.text) for e in episodes] == [("u1", "поправь заголовок")]


def test_artifact_view_context_as_separate_block_keeps_reply(tmp_path: Path) -> None:
    blocks: list[dict[str, object]] = [
        {"type": "text", "text": _ARTIFACT_BLOCK},
        {"type": "text", "text": "поправь тест"},
    ]
    episodes = _episodes(_write(tmp_path, [_blocks("u1", 0, blocks)]))
    assert [e.text for e in episodes] == ["поправь тест"]


def test_text_next_to_image_is_episode(tmp_path: Path) -> None:
    blocks: list[dict[str, object]] = [
        {"type": "text", "text": "смотри"},
        {"type": "image", "source": {}},
    ]
    episodes = _episodes(_write(tmp_path, [_blocks("u1", 0, blocks)]))
    assert [e.text for e in episodes] == ["смотри"]


def test_interrupt_is_signal_of_next_episode(tmp_path: Path) -> None:
    lines = [
        user_line("u1", "делай", _ts(0)),
        _blocks("i1", 1, [{"type": "text", "text": "[Request interrupted by user]"}]),
        user_line("u2", "стоп, не так", _ts(2)),
        user_line("u3", "ещё", _ts(3)),
    ]
    episodes = _episodes(_write(tmp_path, lines))
    assert [(e.uuid, e.signals) for e in episodes] == [
        ("u1", ()),
        ("u2", ("прерывание",)),
        ("u3", ()),
    ]


def test_rejection_without_reason_marks_next_episode(tmp_path: Path) -> None:
    lines = [
        user_line("u1", "делай", _ts(0)),
        _tool_result("r1", 1, REJECT),
        _blocks("i1", 2, [{"type": "text", "text": "[Request interrupted by user for tool use]"}]),
        user_line("u2", "иначе", _ts(3)),
    ]
    episodes = _episodes(_write(tmp_path, lines))
    assert [(e.uuid, e.signals) for e in episodes] == [("u1", ()), ("u2", ("отказ",))]


def test_rejection_with_reason_is_episode(tmp_path: Path) -> None:
    reason = REJECT + "\nTo tell you how to proceed, the user said:\nне трогай конфиг"
    lines = [user_line("u1", "делай", _ts(0)), _tool_result("r1", 1, reason)]
    episodes = _episodes(_write(tmp_path, lines))
    assert [(e.uuid, e.text, e.signals) for e in episodes] == [
        ("u1", "делай", ()),
        ("r1", "не трогай конфиг", ("отказ",)),
    ]


def test_reason_as_separate_text_block(tmp_path: Path) -> None:
    blocks: list[dict[str, object]] = [
        {"type": "tool_result", "content": REJECT, "is_error": True, "tool_use_id": "t"},
        {"type": "text", "text": "лучше через sed"},
    ]
    episodes = _episodes(_write(tmp_path, [_blocks("r1", 0, blocks)]))
    assert [(e.text, e.signals) for e in episodes] == [("лучше через sed", ("отказ",))]


def test_automode_and_rule_denials_are_not_rejections(tmp_path: Path) -> None:
    lines = [
        _tool_result("r1", 0, AUTOMODE),
        _tool_result("r2", 1, RULE),
        user_line("u1", "дальше", _ts(2)),
    ]
    episodes = _episodes(_write(tmp_path, lines))
    assert [(e.uuid, e.signals) for e in episodes] == [("u1", ())]


def test_pasted_content_is_masked(tmp_path: Path) -> None:
    text = 'смотри\n<pasted_content id="3131">\nabcdef\n</pasted_content id="3131">\nвсё'
    episodes = _episodes(_write(tmp_path, [user_line("u1", text, _ts(0))]))
    assert episodes[0].text == "смотри\n[вставка 8 символов]\nвсё"


def test_subagent_sessions_are_not_read(tmp_path: Path) -> None:
    logs = _write(tmp_path, [user_line("u1", "основная", _ts(0))])
    sub = logs / "-work-proj" / "s1" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-1.jsonl").write_text(user_line("x1", "из субагента", _ts(1)) + "\n")
    assert [e.uuid for e in _episodes(logs)] == ["u1"]


def test_worktree_dir_belongs_to_parent_project(tmp_path: Path) -> None:
    wt = "/work/proj/.claude/worktrees/feat-1"
    lines = [
        user_line("u1", "из worktree", _ts(0), cwd=wt),
        json.dumps(
            {
                "type": "user",
                "uuid": "u2",
                "timestamp": _ts(1),
                "message": {"role": "user", "content": "без cwd"},
            }
        ),
    ]
    logs = _write(tmp_path, lines, name="-work-proj--claude-worktrees-feat-1")
    assert [e.project for e in _episodes(logs)] == ["/work/proj", "/work/proj"]


def _bare(uuid: str, n: int) -> dict[str, object]:
    message = {"role": "user", "content": "без cwd"}
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": _ts(n),
        "sessionId": "s1",
        "message": message,
    }


def test_cwd_and_dir_name_give_same_project(tmp_path: Path) -> None:
    # Путь с дефисом не декодируется из имени каталога; проект берётся из cwd соседней записи.
    cwd = "/work/my-proj"
    bare = _bare("u2", 1)
    lines = [user_line("u1", "с cwd", _ts(0), cwd=cwd), json.dumps(bare)]
    logs = _write(tmp_path, lines, name="-work-my-proj")
    assert [e.project for e in _episodes(logs)] == [cwd, cwd]


def test_dir_name_without_any_cwd_is_decoded(tmp_path: Path) -> None:
    bare = _bare("u1", 0)
    logs = _write(tmp_path, [json.dumps(bare)], name="-work-proj--claude-worktrees-f")
    assert [e.project for e in _episodes(logs)] == ["/work/proj"]


def test_clean_reply_per_block_kind() -> None:
    assert clean_reply("<system-reminder>\nnote\n</system-reminder>").text is None
    assert clean_reply("<system-reminder>x</system-reminder>\nответ").text == "ответ"
    cmd = clean_reply(
        "<command-message>p</command-message>\n<command-name>/p</command-name>\n"
        "<command-args>аргумент</command-args>"
    )
    assert (cmd.text, cmd.is_command) == ("/p аргумент", True)
    paste = clean_reply('смотри <pasted_content id="1">секрет</pasted_content id="1">')
    assert paste.text == "смотри [вставка 6 символов]"
    assert clean_reply(_ARTIFACT_BLOCK).text is None
    assert clean_reply(_ARTIFACT_BLOCK + " ответ").text == "ответ"
    assert clean_reply('<artifact-view-context artifact="a"> {"context"').text is None
    assert clean_reply("<task-notification>\n<task-id>x</task-id>").text is None
    assert clean_reply("<local-command-stdout>ok</local-command-stdout>").text is None
    assert clean_reply("This session is being continued from a previous conversation.").text is None
    assert clean_reply("[Request interrupted by user]").is_interrupt


def _skill_lines(args: str) -> list[str]:
    command = f"<command-name>/plain</command-name>\n<command-args>{args}</command-args>"
    return [
        _blocks("i1", 0, [{"type": "text", "text": "[Request interrupted by user]"}]),
        user_line("u1", command, _ts(1)),
    ]


def test_scan_bare_skill_call_is_neutral_without_model(engine: Engine, tmp_path: Path) -> None:
    calls: list[str] = []

    def call(prompt: str) -> str:
        calls.append(prompt)
        raise AssertionError("модель не должна вызываться")

    result = scan(
        engine,
        call,
        logs_dir=_write(tmp_path, _skill_lines("")),
        since=None,
        threads=1,
        classifier_version="v",
        skills_dir=make_skill(tmp_path / "skills", "plain"),
    )
    assert (result.found, result.classified, calls) == (1, 0, [])
    with Session(engine) as s:
        episode = s.scalars(select(EpisodeRow)).one()
        row = s.scalars(select(ClassificationRow)).one()
    assert episode.signals == ["прерывание", "вызов скилла"]
    assert episode.skill == "plain"
    assert row.category == "нейтрально"


def test_scan_skill_call_with_args_sends_only_arguments(engine: Engine, tmp_path: Path) -> None:
    prompts: list[str] = []

    def call(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"category": "инструкция", "gist": "не писать доки"})

    logs = _write(tmp_path, _skill_lines("не пиши доки, только структура"))
    result = scan(
        engine,
        call,
        logs_dir=logs,
        since=None,
        threads=1,
        classifier_version="v",
        skills_dir=make_skill(tmp_path / "skills", "plain"),
    )
    assert result.classified == 1
    (prompt,) = prompts
    assert prompt.endswith("Реплика пользователя:\nне пиши доки, только структура")
    assert "/plain" not in prompt
    assert "вызов скилла" in prompt  # признак виден модели
    again = scan(
        engine,
        call,
        logs_dir=logs,
        since=None,
        threads=1,
        classifier_version="v",
        skills_dir=make_skill(tmp_path / "skills", "plain"),
    )
    assert (again.classified, len(prompts)) == (0, 1)
    with Session(engine) as s:
        assert s.scalars(select(EpisodeRow)).one().skill == "plain"


def test_scan_classifier_context_has_marker(engine: Engine, tmp_path: Path) -> None:
    text = '<pasted_content id="1">секрет</pasted_content id="1">'
    prompts: list[str] = []

    def call(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"category": "запрос", "gist": "g"})

    scan(
        engine,
        call,
        logs_dir=_write(tmp_path, [user_line("u1", text, _ts(0))]),
        since=None,
        threads=1,
        classifier_version="v",
        skills_dir=NO_SKILLS,
    )
    assert len(prompts) == 1
    assert "[вставка 6 символов]" in prompts[0]
    assert "секрет" not in prompts[0]
    with Session(engine) as s:
        assert s.scalars(select(EpisodeRow.text)).one() == "[вставка 6 символов]"


def test_quoted_command_tag_is_plain_episode(tmp_path: Path) -> None:
    text = "почему в логе <command-name>/x</command-name> так"
    records = files_to_records(find_log_files(_write(tmp_path, [user_line("u1", text, _ts(0))])))
    assert [(r.text, r.is_command) for r in records] == [(text, False)]


def test_builtin_command_is_not_an_episode(tmp_path: Path) -> None:
    def command(name: str, args: str) -> str:
        return (
            f"<command-message>{name}</command-message>\n<command-name>/{name}</command-name>\n"
            f"<command-args>{args}</command-args>"
        )

    lines = [
        user_line("u1", command("compact", ""), _ts(0)),
        user_line("u3", command("model", "opus"), _ts(1)),
        user_line("u2", command("py-setup", "спека"), _ts(2)),
    ]
    skills = make_skill(tmp_path / "skills", "py-setup")
    episodes = _episodes(_write(tmp_path, lines), skills)
    assert [(e.text, e.skill, e.signals, e.category) for e in episodes] == [
        ("/py-setup спека", "py-setup", ("вызов скилла",), None),
    ]


def test_project_skill_counts(tmp_path: Path) -> None:
    command = (
        "<command-message>local</command-message>\n<command-name>/local</command-name>\n"
        "<command-args>x</command-args>"
    )
    logs = _write(tmp_path, [user_line("u1", command, _ts(0), cwd=str(tmp_path / "proj"))])
    make_skill(tmp_path / "proj" / ".claude" / "skills", "local")
    (episode,) = _episodes(logs)
    assert episode.skill == "local"
