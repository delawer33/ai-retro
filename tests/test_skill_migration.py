"""Миграция 0004 (ADR 0004): метки «вызов скилла» в эталоне и возврат эпизодов в разметку."""

import json
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text
from typer.testing import CliRunner

from ai_retro.cli import app
from ai_retro.db.session import MIGRATIONS_DIR, make_engine
from ai_retro.label.service import returned_count

OLD = "вызов скилла"
TS = "2026-09-25 11:00:00"


def _upgrade(engine: Engine, revision: str) -> None:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    with engine.begin() as connection:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, revision)


def _old_database(engine: Engine) -> None:
    _upgrade(engine, "0002")
    with engine.begin() as c:
        for uuid, body in [("bare", "/feat"), ("args", "/feat не пиши доки"), ("plain", "ок")]:
            c.execute(
                text(
                    "INSERT INTO episodes (uuid, conversation_id, project, timestamp, text, signals)"
                    " VALUES (:u, 'c', 'p', :ts, :t, '[]')"
                ),
                {"u": uuid, "ts": TS, "t": body},
            )
            category = OLD if uuid != "plain" else "нейтрально"
            c.execute(
                text(
                    "INSERT INTO classifications (episode_uuid, category, gist, classifier_version)"
                    " VALUES (:u, :c, 'g', 'v0')"
                ),
                {"u": uuid, "c": category},
            )
            c.execute(
                text(
                    "INSERT INTO gold_labels (episode_uuid, category, split, labeled_at, is_relabel)"
                    " VALUES (:u, :c, 'dev', :ts, 0)"
                ),
                {"u": uuid, "c": category, "ts": TS},
            )
            c.execute(
                text("INSERT INTO gold_sample (episode_uuid, split) VALUES (:u, 'dev')"),
                {"u": uuid},
            )


def test_migration_translates_gold_labels(tmp_path: Path) -> None:
    engine = make_engine(tmp_path / "old.db")
    _old_database(engine)
    _upgrade(engine, "head")
    with engine.connect() as c:
        gold = {u: k for u, k in c.execute(text("SELECT episode_uuid, category FROM gold_labels"))}
        skills = {u: k for u, k in c.execute(text("SELECT uuid, skill FROM episodes"))}
        old_classifications = c.execute(
            text("SELECT count(*) FROM classifications WHERE category = :o"), {"o": OLD}
        ).scalar()
        signals = c.execute(text("SELECT signals FROM episodes WHERE uuid = 'bare'")).scalar()
    assert gold == {"bare": "нейтрально", "plain": "нейтрально"}  # метка «args» снята
    assert skills == {"bare": "feat", "args": "feat", "plain": None}
    assert old_classifications == 0
    assert json.loads(str(signals)) == [OLD]
    assert returned_count(engine) == 1


def test_label_reports_returned_episodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "old.db"
    _old_database(make_engine(db))
    _upgrade(make_engine(db), "head")
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(db))
    result = CliRunner().invoke(app, ["label"], input="q")
    assert result.exit_code == 0, result.output
    assert "Вернулось в разметку после отмены категории «вызов скилла»: 1" in result.output
