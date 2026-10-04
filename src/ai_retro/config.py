from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_RETRO_", env_file=".env", extra="ignore")

    logs_dir: Path = Path.home() / ".claude" / "projects"
    db_path: Path = Path.home() / ".local" / "share" / "ai-retro" / "ai-retro.db"
    model: str = "haiku"
    threads: int = 3
    domain_rules: str = "all/coll=учёба;all/projects=личные проекты;Documents/avito=работа"
    home: Path = Path.home()
