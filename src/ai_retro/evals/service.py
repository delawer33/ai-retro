import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from ai_retro.classify.service import CATEGORIES, CLASSIFIER_VERSION, classify_many
from ai_retro.db.models import ClassificationRow, EpisodeRow, EvalRunRow, GoldLabelRow
from ai_retro.errors import EmptyGoldSetError
from ai_retro.evals.baseline import baseline_category
from ai_retro.evals.criteria import Check, stop_criteria
from ai_retro.evals.metrics import Metrics, Pair, agreement, class_deltas, compute_metrics
from ai_retro.llm import ModelCall

DEV = "dev"
TEST = "test"
PROMPT_FILE = Path(__file__).resolve().parents[1] / "classify" / "service.py"


@dataclass(frozen=True)
class Disagreement:
    uuid: str
    human: str
    model: str
    text: str


@dataclass(frozen=True)
class EvalResult:
    split: str
    n: int
    model: Metrics
    baseline: Metrics
    ceiling: float | None
    ceiling_n: int
    deltas: dict[str, dict[str, float | None]] | None  # против прошлого прогона того же split
    disagreements: list[Disagreement]
    checks: list[Check]


def prompt_git_sha(path: Path = PROMPT_FILE) -> str:
    """Короткий git sha последнего коммита, тронувшего промпт; `unknown` вне git."""
    try:
        proc = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", path.name],
            cwd=path.parent,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _labels(session: Session) -> tuple[dict[str, GoldLabelRow], dict[str, GoldLabelRow]]:
    """(последняя обычная метка, последняя повторная) по эпизоду."""
    gold: dict[str, GoldLabelRow] = {}
    relabels: dict[str, GoldLabelRow] = {}
    for row in session.scalars(select(GoldLabelRow).order_by(GoldLabelRow.id)):
        (relabels if row.is_relabel else gold)[row.episode_uuid] = row
    return gold, relabels


def _ceiling_pairs(session: Session) -> list[Pair]:
    """Первая метка человека против повторной вслепую."""
    first: dict[str, str] = {}
    for row in session.scalars(
        select(GoldLabelRow).where(GoldLabelRow.is_relabel.is_(False)).order_by(GoldLabelRow.id)
    ):
        first.setdefault(row.episode_uuid, row.category)
    _, relabels = _labels(session)
    return [(first[u], r.category) for u, r in relabels.items() if u in first]


def _model_categories(
    session: Session, episodes: list[EpisodeRow], call: ModelCall, *, threads: int, refresh: bool
) -> dict[str, str]:
    """Категории из базы; недостающие (или все при refresh) классифицируются заново."""
    categories: dict[str, str] = {}
    for row in session.scalars(
        select(ClassificationRow)
        .where(ClassificationRow.classifier_version == CLASSIFIER_VERSION)
        .order_by(ClassificationRow.id)
    ):
        categories[row.episode_uuid] = row.category
    todo = [(e.uuid, e.text) for e in episodes if refresh or e.uuid not in categories]
    for uuid, result in classify_many(call, todo, threads=threads).items():
        session.add(
            ClassificationRow(
                episode_uuid=uuid,
                category=result.category,
                gist=result.gist,
                classifier_version=CLASSIFIER_VERSION,
            )
        )
        categories[uuid] = result.category
    session.commit()
    return categories


def run_eval(
    engine: Engine,
    call: ModelCall,
    *,
    split: str = DEV,
    refresh: bool = False,
    threads: int = 1,
    model: str,
    claude_version: str,
    prompt_sha: str,
    now: datetime | None = None,
) -> EvalResult:
    with Session(engine) as session:
        gold, _ = _labels(session)
        uuids = [u for u, row in gold.items() if row.split == split]
        if not uuids:
            raise EmptyGoldSetError(split)
        episodes = list(
            session.scalars(
                select(EpisodeRow).where(EpisodeRow.uuid.in_(uuids)).order_by(EpisodeRow.timestamp)
            )
        )
        predicted = _model_categories(session, episodes, call, threads=threads, refresh=refresh)

        model_pairs = [(gold[e.uuid].category, predicted[e.uuid]) for e in episodes]
        baseline_pairs = [
            (gold[e.uuid].category, baseline_category(e.text, e.signals)) for e in episodes
        ]
        ceiling_pairs = _ceiling_pairs(session)
        model_metrics = compute_metrics(model_pairs, CATEGORIES)
        baseline_metrics = compute_metrics(baseline_pairs, CATEGORIES)
        ceiling = agreement(ceiling_pairs)

        previous = session.scalars(
            select(EvalRunRow).where(EvalRunRow.split == split).order_by(EvalRunRow.id.desc())
        ).first()
        deltas = (
            class_deltas(model_metrics.to_dict()["classes"], previous.metrics["model"]["classes"])
            if previous is not None
            else None
        )

        record: dict[str, Any] = {
            "model": model_metrics.to_dict(),
            "baseline": baseline_metrics.to_dict(),
            "ceiling": {"agreement": ceiling, "n": len(ceiling_pairs)},
        }
        disagreements = [
            Disagreement(uuid=e.uuid, human=h, model=m, text=e.text)
            for e, (h, m) in zip(episodes, model_pairs, strict=True)
            if h != m
        ]

        session.add(
            EvalRunRow(
                started_at=now or datetime.now(UTC),
                classifier_version=f"{CLASSIFIER_VERSION}@{prompt_sha}",
                claude_version=claude_version,
                model=model,
                split=split,
                metrics=record,
            )
        )
        session.commit()

    return EvalResult(
        split=split,
        n=len(episodes),
        model=model_metrics,
        baseline=baseline_metrics,
        ceiling=ceiling,
        ceiling_n=len(ceiling_pairs),
        deltas=deltas,
        disagreements=disagreements,
        checks=stop_criteria(model_metrics, baseline_metrics, ceiling),
    )
