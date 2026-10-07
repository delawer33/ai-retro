from ai_retro.llm import claude_args


def test_claude_args_match_adr_0002() -> None:
    args = claude_args(system_prompt="SP", json_schema={"type": "object"}, model="haiku")
    assert args == [
        "claude", "-p", "--safe-mode", "--system-prompt", "SP", "--tools", "",
        "--model", "haiku", "--json-schema", '{"type": "object"}',
        "--output-format", "json", "--no-session-persistence",
    ]  # fmt: skip


def test_extract_payload_flags_unrecognized_model() -> None:
    import json

    import pytest

    from ai_retro.errors import UnrecognizedModelError
    from ai_retro.llm import _extract_payload

    stdout = json.dumps({"is_error": True, "result": "There's an issue with the selected model"})
    with pytest.raises(UnrecognizedModelError, match="несуществующая"):
        _extract_payload(stdout, "несуществующая")
