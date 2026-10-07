from pathlib import Path

import pytest

from ai_retro.config import Settings


def test_tilde_expands_in_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/u")
    monkeypatch.setenv("AI_RETRO_LOGS_DIR", "~/.claude/projects")
    monkeypatch.setenv("AI_RETRO_DB_PATH", "~/.local/share/ai-retro/ai-retro.db")
    monkeypatch.setenv("AI_RETRO_HOME", "~")
    settings = Settings()
    assert settings.logs_dir == Path("/home/u/.claude/projects")
    assert settings.db_path == Path("/home/u/.local/share/ai-retro/ai-retro.db")
    assert settings.home == Path("/home/u")


def test_env_example_keeps_tilde() -> None:
    example = Path(__file__).parents[1] / ".env.example"
    assert "AI_RETRO_LOGS_DIR=~/.claude/projects" in example.read_text(encoding="utf-8")
