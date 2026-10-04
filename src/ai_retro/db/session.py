from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


def make_engine(db_path: Path) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def migrate(engine: Engine) -> None:
    """Приводит схему к head. Схема создаётся только миграциями."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    with engine.begin() as connection:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")
