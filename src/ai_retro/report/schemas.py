from dataclasses import dataclass
from datetime import datetime

TARGET_PROJECT = "project"
TARGET_GLOBAL = "global"
TARGET_SKILL = "skill"
PROPOSED = "proposed"
STALE = "stale"


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
    # скилл, с которым связан эпизод: вызван им или идёт в беседе после его вызова
    skill: str | None = None


@dataclass(frozen=True)
class Target:
    """Куда пойдёт предложение. `file` пуст у нового скилла: у него файла ещё нет."""

    kind: str  # TARGET_PROJECT | TARGET_GLOBAL | TARGET_SKILL
    file: str


@dataclass(frozen=True)
class Proposal:
    """Черновик правки: ровно этот текст потом применит `apply` (этап 2)."""

    target_file: str  # пусто — предложение только текстом, «новый скилл»
    section: str
    text: str
    rationale: str
    status: str = PROPOSED


@dataclass(frozen=True)
class Pattern:
    title: str
    episodes: tuple[GistItem, ...]
    proposal: Proposal | None = None

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
