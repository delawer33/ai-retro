# Ai Retro

Анализ сессий Claude Code и поиск повторяющихся паттернов

Глоссарий: `CONTEXT.md` — его словами пишутся код, тесты и доки. Решения: `docs/adr/` — перед
изменением области читай ADR, которые её касаются; пиши новый, когда делаешь выбор, который
компетентный инженер мог бы сделать иначе.

## Команды

```bash
uv sync                          # зависимости + venv (dev-группа включена)
uv run ai-retro --help        # сам инструмент; команды объявлены в src/ai_retro/cli.py
uv run pytest                    # тесты; один: uv run pytest tests/test_x.py -k name
uv run ruff check . && uv run ruff format --check .
uv run mypy                      # strict; конфиг в pyproject.toml
uv run pre-commit install        # один раз на клон
```

## Структура

```
src/ai_retro/        package by feature: один каталог на доменный срез
  cli.py                typer-приложение; команда = тонкая функция, которая зовёт срез (создаёт первый срез)
  config.py             Settings (pydantic-settings), всё через env
  <feature>/            service.py, schemas.py, models.py — только то, что нужно срезу
tests/                  pytest; команды тестируются через typer.testing.CliRunner, логика — напрямую
migrations/             изменения схемы (Alembic); create_all в коде приложения не бывает
docs/adr/               решения
```

## Конвенции

- Команда в `cli.py` только разбирает аргументы и печатает результат; вся логика — в срезе и
  тестируется без CLI.
- Сервисы — функции, принимающие зависимости явно; классы — только при реальном состоянии или
  полиморфизме. Абстракции (Protocol/ABC) — только там, где вторая реализация есть или предрешена.
- Конфигурация только через env (`config.py`); `.env` не коммитится, `.env.example` — всегда.
- Ожидаемые ошибки — типизированное доменное исключение с сообщением, по которому пользователь
  может действовать; `except Exception` не ловить. CLI переводит его в код выхода ≠ 0 и одну строку.
- `pathlib.Path`, не `os.path`; `X | None`; фичи Python 3.13 использовать можно.
- Код пишется сразу под конфиг ruff (line-length 100), чтобы формат на pre-commit был no-op.
- Conventional commits (`feat:`, `fix:`, `chore:`, ...). Trunk-based, короткие ветки, зелёный CI
  перед мержем.
