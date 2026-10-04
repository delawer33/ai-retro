from ai_retro.llm import claude_args


def test_claude_args_match_adr_0002() -> None:
    args = claude_args(system_prompt="SP", json_schema={"type": "object"}, model="haiku")
    assert args == [
        "claude", "-p", "--safe-mode", "--system-prompt", "SP", "--tools", "",
        "--model", "haiku", "--json-schema", '{"type": "object"}',
        "--output-format", "json", "--no-session-persistence",
    ]  # fmt: skip
