import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ai_retro.parsing.conversations import records_to_conversations
from ai_retro.parsing.episodes import conversations_to_episodes
from ai_retro.parsing.files import files_to_records, find_log_files
from ai_retro.parsing.schemas import Conversation, EpisodeDraft
from tests.conftest import NO_SKILLS, user_line


def test_files_to_records_skips_noise(logs_dir: Path) -> None:
    records = files_to_records(find_log_files(logs_dir))
    assert [r.uuid for r in records] == ["u1", "a1", "u2", "t1", "m1"]
    assert records[0].text == "сделай X"
    assert records[1].text is None
    assert records[3].text is None  # tool_result
    assert records[4].text is None  # isMeta


def test_one_file_is_one_conversation(logs_dir: Path) -> None:
    conversations = records_to_conversations(files_to_records(find_log_files(logs_dir)))
    assert len(conversations) == 1
    assert conversations[0].id == "s1"


def test_episodes_are_user_replies(logs_dir: Path) -> None:
    conversations = records_to_conversations(files_to_records(find_log_files(logs_dir)))
    episodes = conversations_to_episodes(conversations, skills_dir=NO_SKILLS)
    assert [e.uuid for e in episodes] == ["u1", "u2"]
    assert episodes[0].conversation_id == "s1"
    assert episodes[0].project == "/work/proj"


# --- беседы из нескольких файлов и rewind (ADR 0001) ---


def _write(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _u(uuid: str, text: str, minute: int, parent: str | None) -> str:
    return user_line(uuid, text, f"2026-09-25T11:{minute:02d}:00.000Z", parent=parent)


def _a(
    uuid: str,
    text: str,
    minute: int,
    parent: str,
    *,
    blocks: Sequence[Mapping[str, object]] | None = None,
) -> str:
    content = blocks if blocks is not None else [{"type": "text", "text": text}]
    return json.dumps(
        {
            "type": "assistant",
            "uuid": uuid,
            "parentUuid": parent,
            "timestamp": f"2026-09-25T11:{minute:02d}:30.000Z",
            "sessionId": "s",
            "message": {"role": "assistant", "content": content},
        }
    )


def _tool_result(uuid: str, minute: int, parent: str) -> str:
    return json.dumps(
        {
            "type": "user",
            "uuid": uuid,
            "parentUuid": parent,
            "timestamp": f"2026-09-25T11:{minute:02d}:40.000Z",
            "sessionId": "s",
            "message": {"role": "user", "content": [{"type": "tool_result", "content": "r"}]},
        }
    )


def _episodes(project: Path) -> dict[str, EpisodeDraft]:
    conversations = records_to_conversations(files_to_records(find_log_files(project.parent)))
    return {e.uuid: e for e in conversations_to_episodes(conversations, skills_dir=NO_SKILLS)}


def _conversations(project: Path) -> list[Conversation]:
    return records_to_conversations(files_to_records(find_log_files(project.parent)))


def test_rewind_into_new_file(tmp_path: Path) -> None:
    project = tmp_path / "logs" / "-p"
    _write(
        project / "old.jsonl",
        [
            _u("u1", "первая", 0, None),
            _a("a1", "ответ один", 0, "u1"),
            _u("u2", "старая вторая", 1, "a1"),
            _a("a2", "ответ на старую", 1, "u2"),
            _u("u3", "старая третья", 2, "a2"),
        ],
    )
    _write(
        project / "new.jsonl",
        [
            _u("u1", "первая", 0, None),
            _a("a1", "ответ один", 0, "u1"),
            _u("n2", "новая вторая", 5, "a1"),
            _a("b2", "ответ на новую", 5, "n2"),
        ],
    )
    conversations = _conversations(project)
    assert len(conversations) == 1
    episodes = _episodes(project)
    assert sorted(episodes) == ["n2", "u1", "u2", "u3"]  # по одному эпизоду на реплику
    assert episodes["n2"].signals == ("rewind",)
    assert all(episodes[u].signals == () for u in ("u1", "u2", "u3"))
    context = episodes["n2"].context or ""
    assert "старая вторая" in context
    assert "ответ на старую" in context
    assert "старая третья" not in context
    assert episodes["u2"].context == "ответ один"  # не rewind: последний ответ агента


def test_rewind_context_skips_one_line_preamble(tmp_path: Path) -> None:
    project = tmp_path / "logs" / "-p"
    tool = [{"type": "tool_use", "id": "t", "name": "Read", "input": {}}]
    _write(
        project / "old.jsonl",
        [
            _u("u1", "первая", 0, None),
            _a("a1", "ответ один", 0, "u1"),
            _u("u2", "старая вторая", 1, "a1"),
            _a("a2", "Посмотрю журнал прогона.", 1, "u2"),
            _a("a3", "", 1, "a2", blocks=tool),
            _tool_result("t1", 2, "a3"),
            _a("a4", "В журнале видно: упал шаг сборки из-за версии.", 2, "t1"),
            _u("u3", "старая третья", 3, "a4"),
        ],
    )
    _write(
        project / "new.jsonl",
        [
            _u("u1", "первая", 0, None),
            _a("a1", "ответ один", 0, "u1"),
            _u("n2", "новая вторая", 5, "a1"),
        ],
    )
    context = _episodes(project)["n2"].context or ""
    assert "Посмотрю журнал прогона." in context
    assert "упал шаг сборки из-за версии" in context
    assert "старая третья" not in context


def test_chain_of_two_rewinds(tmp_path: Path) -> None:
    project = tmp_path / "logs" / "-p"
    _write(
        project / "f1.jsonl",
        [_u("u1", "корень", 0, None), _a("a1", "ок", 0, "u1"), _u("x2", "ветка один", 1, "a1")],
    )
    _write(
        project / "f2.jsonl",
        [
            _u("u1", "корень", 0, None),
            _a("a1", "ок", 0, "u1"),
            _u("y2", "ветка два", 5, "a1"),
            _a("b2", "ответ два", 5, "y2"),
            _u("y3", "второй шаг", 6, "b2"),
        ],
    )
    _write(
        project / "f3.jsonl",
        [
            _u("u1", "корень", 0, None),
            _a("a1", "ок", 0, "u1"),
            _u("y2", "ветка два", 5, "a1"),
            _a("b2", "ответ два", 5, "y2"),
            _u("z3", "ветка три", 9, "b2"),
        ],
    )
    episodes = _episodes(project)
    assert sorted(episodes) == ["u1", "x2", "y2", "y3", "z3"]
    assert {u for u, e in episodes.items() if e.signals} == {"y2", "z3"}
    assert "ветка один" in (episodes["y2"].context or "")
    assert "второй шаг" in (episodes["z3"].context or "")


def test_parallel_tool_calls_are_not_rewind(tmp_path: Path) -> None:
    project = tmp_path / "logs" / "-p"
    tool = [{"type": "tool_use", "id": "t", "name": "Read", "input": {}}]
    _write(
        project / "s.jsonl",
        [
            _u("u1", "прочти оба", 0, None),
            _a("a1", "", 0, "u1", blocks=tool),
            _a("a2", "", 0, "u1", blocks=tool),
            _tool_result("t1", 1, "a1"),
            _tool_result("t2", 1, "a2"),
            _a("a3", "готово", 2, "t1"),
            _u("u2", "дальше", 3, "a3"),
        ],
    )
    episodes = _episodes(project)
    assert sorted(episodes) == ["u1", "u2"]
    assert all(e.signals == () for e in episodes.values())


def test_resume_is_not_rewind_and_has_no_duplicates(tmp_path: Path) -> None:
    project = tmp_path / "logs" / "-p"
    base = [_u("u1", "первая", 0, None), _a("a1", "ок", 0, "u1"), _u("u2", "вторая", 1, "a1")]
    _write(project / "old.jsonl", base)
    _write(
        project / "resumed.jsonl", [*base, _a("a2", "ещё", 5, "u2"), _u("u3", "третья", 6, "a2")]
    )
    conversations = _conversations(project)
    assert len(conversations) == 1
    assert len(conversations[0].records) == len({r.uuid for r in conversations[0].records}) == 5
    episodes = _episodes(project)
    assert sorted(episodes) == ["u1", "u2", "u3"]
    assert all(e.signals == () for e in episodes.values())


def test_unrelated_files_are_separate_conversations(tmp_path: Path) -> None:
    project = tmp_path / "logs" / "-p"
    _write(project / "a.jsonl", [_u("u1", "раз", 0, None)])
    _write(project / "b.jsonl", [_u("v1", "два", 1, None)])
    assert len(_conversations(project)) == 2
