from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class GistItem:
    """Эпизод поправки или инструкции, пошедший в группировку по сутям."""

    uuid: str
    conversation_id: str
    project: str
    domain: str
    category: str
    gist: str
    text: str
    timestamp: datetime


@dataclass(frozen=True)
class Pattern:
    title: str
    episodes: tuple[GistItem, ...]

    @property
    def conversations(self) -> frozenset[str]:
        return frozenset(e.conversation_id for e in self.episodes)

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({e.domain for e in self.episodes}))

    @property
    def projects(self) -> tuple[str, ...]:
        return tuple(sorted({e.project for e in self.episodes}))


@dataclass(frozen=True)
class ReportData:
    since: datetime | None
    # область -> категория -> число эпизодов
    distribution: dict[str, dict[str, int]]
    patterns: tuple[Pattern, ...]  # по убыванию числа эпизодов
