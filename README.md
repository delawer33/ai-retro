# Ai Retro

Анализ сессий Claude Code и поиск повторяющихся паттернов

## Запуск

```bash
cp .env.example .env
uv sync
uv run ai-retro --help
```

## Разработка

```bash
uv run pre-commit install
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Заметки для агентов — в `CLAUDE.md`; словарь домена — в `CONTEXT.md`; решения — в `docs/adr/`.
