from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_RETRO_", env_file=".env", extra="ignore")

    logs_dir: Path = Path.home() / ".claude" / "projects"
    db_path: Path = Path.home() / ".local" / "share" / "ai-retro" / "ai-retro.db"
    model: str = "haiku"
    threads: int = 3
    context_chars: int = 2000

    domain_rules: str = "work=работа;projects=личные проекты;study=учёба"
    home: Path = Path.home()
    skills_dir: Path = Path.home() / ".claude" / "skills"

    @field_validator("logs_dir", "db_path", "home", "skills_dir")
    @classmethod
    def _expand_tilde(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("domain_rules")
    @classmethod
    def _expand_rule_prefixes(cls, value: str) -> str:
        """`~/x=учёба` → абсолютный префикс; остальное правило не трогаем (разбор — parse_rules)."""
        parts = []
        for part in value.split(";"):
            prefix, sep, domain = part.rpartition("=")
            if sep and prefix.strip().startswith("~"):
                part = f"{Path(prefix.strip()).expanduser()}={domain}"
            parts.append(part)
        return ";".join(parts)
