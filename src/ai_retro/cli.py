from datetime import UTC, datetime
from typing import Annotated

import typer

from ai_retro.classify.service import (
    CATEGORIES,
    JSON_SCHEMA,
    SYSTEM_PROMPT,
    classifier_version,
)
from ai_retro.config import Settings
from ai_retro.db.session import make_engine, migrate
from ai_retro.domain.service import JSON_SCHEMA as DOMAIN_SCHEMA
from ai_retro.domain.service import SYSTEM_PROMPT as DOMAIN_PROMPT
from ai_retro.domain.service import parse_rules
from ai_retro.errors import AiRetroError
from ai_retro.evals.render import render
from ai_retro.evals.service import DEV, TEST, ensure_gold, run_eval
from ai_retro.label.service import (
    ensure_sample,
    pending_items,
    pending_relabel_items,
    render_item,
    save_label,
)
from ai_retro.llm import claude_version, make_claude_call
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
        domain_rules = parse_rules(settings.domain_rules, home=settings.home)
        domain_call = make_claude_call(
            system_prompt=DOMAIN_PROMPT, json_schema=DOMAIN_SCHEMA, model=settings.model
        )
        engine = make_engine(settings.db_path)
        migrate(engine)
        result = scan(
            engine,
            call,
            logs_dir=settings.logs_dir,
            since=since_dt,
            threads=settings.threads,
            classifier_version=classifier_version(settings.model),
            context_chars=settings.context_chars,
            announce=lambda n: typer.echo(f"Эпизодов к классификации (уйдёт в модель): {n}"),
            domain_rules=domain_rules,
            domain_call=domain_call,
        )
    except AiRetroError as exc:
        typer.echo(f"Ошибка: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Эпизодов найдено: {result.found}, классифицировано: {result.classified}")


@app.command("label")
def label_command(
    relabel: Annotated[
        bool, typer.Option(help="Повторная разметка вслепую: 20 размеченных эпизодов заново.")
    ] = False,
    extend: Annotated[bool, typer.Option(help="Добавить в выборку новые эпизоды.")] = False,
) -> None:
    """Разметить эпизоды выборки в терминале: так собирается эталон."""
    settings = Settings()
    try:
        engine = make_engine(settings.db_path)
        migrate(engine)
        if relabel:
            items = pending_relabel_items(engine)
        else:
            added = ensure_sample(engine, extend=extend)
            if added:
                typer.echo(f"В выборку добавлено эпизодов: {added}")
            items = pending_items(engine)
        if not items:
            typer.echo("Размечать нечего.")
            return
        keys = {str(i): c for i, c in enumerate(CATEGORIES, start=1)}
        legend = "  ".join(f"{k}={c}" for k, c in keys.items()) + "  s=пропустить  q=выход"
        done = 0
        for number, item in enumerate(items, start=1):
            typer.echo(render_item(item, number=number, total=len(items)))
            typer.echo(legend)
            while True:
                try:
                    key = typer.getchar()
                except (EOFError, KeyboardInterrupt):
                    key = "q"
                if key in keys or key in ("s", "q", "\x03", "\x04", ""):
                    break
            if key in ("q", "\x03", "\x04", ""):
                typer.echo(f"Прервано. Размечено в этом запуске: {done}.")
                return
            if key == "s":
                continue
            save_label(engine, item.uuid, keys[key], relabel=relabel, now=datetime.now(UTC))
            done += 1
    except AiRetroError as exc:
        typer.echo(f"Ошибка: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Готово. Размечено в этом запуске: {done}.")


@app.command("eval")
def eval_command(
    test: Annotated[bool, typer.Option("--test", help="Гнать test вместо dev.")] = False,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Заново классифицировать эпизоды эталона.")
    ] = False,
) -> None:
    """Качество классификатора на эталоне: метрики, бейзлайн, потолок, стоп-критерий."""
    settings = Settings()
    split = TEST if test else DEV
    try:
        engine = make_engine(settings.db_path)
        migrate(engine)
        ensure_gold(engine, split)
        call = make_claude_call(
            system_prompt=SYSTEM_PROMPT, json_schema=JSON_SCHEMA, model=settings.model
        )
        result = run_eval(
            engine,
            call,
            split=split,
            refresh=refresh,
            threads=settings.threads,
            model=settings.model,
            claude_version=claude_version(),
            context_chars=settings.context_chars,
        )
    except AiRetroError as exc:
        typer.echo(f"Ошибка: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(render(result))
