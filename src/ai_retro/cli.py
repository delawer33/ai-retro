from datetime import UTC, datetime
from typing import Annotated

import typer

from ai_retro.classify.service import JSON_SCHEMA, SYSTEM_PROMPT
from ai_retro.config import Settings
from ai_retro.db.session import make_engine, migrate
from ai_retro.errors import AiRetroError
from ai_retro.llm import make_claude_call
from ai_retro.scan.service import parse_since, scan

app = typer.Typer(no_args_is_help=True, help="Анализ сессий Claude Code.")


@app.callback()
def main() -> None:
    """Анализ сессий Claude Code и поиск повторяющихся паттернов."""


@app.command("scan")
def scan_command(
    since: Annotated[str | None, typer.Option(help="Окно, напр. 12h, 1d, 2w.")] = None,
) -> None:
    """Разобрать логи в эпизоды и классифицировать новые."""
    settings = Settings()
    try:
        since_dt = parse_since(since, now=datetime.now(UTC)) if since else None
        call = make_claude_call(
            system_prompt=SYSTEM_PROMPT, json_schema=JSON_SCHEMA, model=settings.model
        )
        engine = make_engine(settings.db_path)
        migrate(engine)
        result = scan(
            engine, call, logs_dir=settings.logs_dir, since=since_dt, threads=settings.threads
        )
    except AiRetroError as exc:
        typer.echo(f"Ошибка: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Эпизодов найдено: {result.found}, классифицировано: {result.classified}")
