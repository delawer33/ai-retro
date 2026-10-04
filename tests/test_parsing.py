from pathlib import Path

from ai_retro.parsing.conversations import records_to_conversations
from ai_retro.parsing.episodes import conversations_to_episodes
from ai_retro.parsing.files import files_to_records, find_log_files


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
    episodes = conversations_to_episodes(conversations)
    assert [e.uuid for e in episodes] == ["u1", "u2"]
    assert episodes[0].conversation_id == "s1"
    assert episodes[0].project == "/work/proj"
