from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from ai_retro.classify.service import (
    CATEGORIES,
    EpisodeInput,
    build_prompt,
    classifier_version,
    classify_many,
    latest_classifications,
)
from ai_retro.db.models import ClassificationRow, EpisodeRow, EvalRunRow, GoldLabelRow
from ai_retro.errors import ClassifierSchemaError, EmptyGoldSetError
from ai_retro.evals.baseline import baseline_category
from ai_retro.evals.criteria import Check, stop_criteria
from ai_retro.evals.metrics import Metrics, Pair, class_deltas, compute_metrics
from ai_retro.llm import ModelCall
from ai_retro.parsing.episodes import NEUTRAL, skill_arguments

DEV = "dev"
TEST = "test"


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


def _episode_input(episode: EpisodeRow) -> EpisodeInput:
    # Те же правила, что у scan: при rewind контекст — начало брошенной ветки.
    rewind = "rewind" in episode.signals
    return EpisodeInput(
        text=skill_arguments(episode.text) if episode.skill else episode.text,
        signals=episode.signals,
        agent_reply=None if rewind else episode.context,
        abandoned_branch=episode.context if rewind else None,
    )


def _is_bare_call(episode: EpisodeRow) -> bool:
    return bool(episode.skill) and not skill_arguments(episode.text)


def _model_categories(
    session: Session,
    episodes: list[EpisodeRow],
    call: ModelCall,
    *,
    version: str,
    threads: int,
    refresh: bool,
    context_chars: int,
) -> dict[str, str]:
    """Категории из базы; недостающие (или все при refresh) классифицируются заново."""
    categories: dict[str, str] = {}
    for uuid, row in latest_classifications(session, classifier_version=version).items():
        categories[uuid] = row.category
    for e in episodes:
        if _is_bare_call(e):
            categories[e.uuid] = NEUTRAL  # голый вызов скилла в модель не идёт
    todo = [
        (e.uuid, build_prompt(_episode_input(e), context_chars=context_chars))
        for e in episodes
        if (refresh or e.uuid not in categories) and not _is_bare_call(e)
    ]
    outcome = classify_many(call, todo, threads=threads)
    # Одна строка на (эпизод, версия): прежнюю заменяем, а не дублируем (ADR 0010).
    session.execute(
        delete(ClassificationRow).where(
            ClassificationRow.classifier_version == version,
            ClassificationRow.episode_uuid.in_(outcome.results),
        )
    )
    for uuid, result in outcome.results.items():
        session.add(
            ClassificationRow(
                episode_uuid=uuid,
                category=result.category,
                gist=result.gist,
                classifier_version=version,
            )
        )
        categories[uuid] = result.category
    session.commit()
    if outcome.error is not None:
        raise outcome.error
    if outcome.unclassified:
        raise ClassifierSchemaError(
            f"Ответ не по схеме на повторе у {len(outcome.unclassified)} эпизодов: "
            "оценка без них исказит метрики. Повтори запуск."
        )
    return categories


def ensure_gold(engine: Engine, split: str) -> None:
    """Падает с подсказкой про `label`, если в split нет эталона — до обращения к модели."""
    with Session(engine) as session:
        gold, _ = _labels(session)
        if not any(row.split == split for row in gold.values()):
            raise EmptyGoldSetError(split)


def run_eval(
    engine: Engine,
    call: ModelCall,
    *,
    split: str = DEV,
    refresh: bool = False,
    threads: int = 1,
    model: str,
    claude_version: str,
    context_chars: int = 2000,
    now: datetime | None = None,
) -> EvalResult:
    version = classifier_version(model)
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
        predicted = _model_categories(
            session,
            episodes,
            call,
            version=version,
            threads=threads,
            refresh=refresh,
            context_chars=context_chars,
        )

        model_pairs = [(gold[e.uuid].category, predicted[e.uuid]) for e in episodes]
        baseline_pairs = [
            (gold[e.uuid].category, baseline_category(e.text, e.signals)) for e in episodes
        ]
        ceiling_pairs = _ceiling_pairs(session)
        model_metrics = compute_metrics(model_pairs, CATEGORIES)
        baseline_metrics = compute_metrics(baseline_pairs, CATEGORIES)
        ceiling_metrics = compute_metrics(ceiling_pairs, CATEGORIES)
        ceiling = ceiling_metrics.agreement

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
            "ceiling": {"agreement": ceiling, "n": len(ceiling_pairs), **ceiling_metrics.to_dict()},
        }
        disagreements = [
            Disagreement(uuid=e.uuid, human=h, model=m, text=e.text)
            for e, (h, m) in zip(episodes, model_pairs, strict=True)
            if h != m
        ]

        session.add(
            EvalRunRow(
                started_at=now or datetime.now(UTC),
                classifier_version=version,
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
