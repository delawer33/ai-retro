import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from ai_retro.classify.service import CATEGORIES, classifier_version
from ai_retro.cli import app
from ai_retro.db.models import ClassificationRow, EpisodeRow, EvalRunRow, GoldLabelRow
from ai_retro.errors import EmptyGoldSetError
from ai_retro.evals.baseline import baseline_category
from ai_retro.evals.criteria import stop_criteria
from ai_retro.evals.metrics import class_deltas, compute_metrics
from ai_retro.evals.render import render
from ai_retro.evals.service import EvalResult, run_eval

T0 = datetime(2026, 9, 25, tzinfo=UTC)


def _fail(_: str) -> str:
    raise AssertionError("модель не должна вызываться")


def _add(
    engine: Engine,
    uuid: str,
    human: str,
    model: str | None,
    *,
    text: str = "реплика",
    split: str = "dev",
    signals: list[str] | None = None,
) -> None:
    with Session(engine) as s:
        s.add(
            EpisodeRow(
                uuid=uuid,
                conversation_id="c",
                project="p",
                timestamp=T0,
                text=text,
                signals=signals or [],
            )
        )
        s.flush()
        s.add(GoldLabelRow(episode_uuid=uuid, category=human, split=split, labeled_at=T0))
        if model:
            s.add(
                ClassificationRow(
                    episode_uuid=uuid,
                    category=model,
                    gist="g",
                    classifier_version=classifier_version("haiku"),
                )
            )
        s.commit()


def _run(engine: Engine, split: str = "dev") -> EvalResult:
    return run_eval(
        engine,
        _fail,
        model="haiku",
        claude_version="9.9",
        split=split,
    )


def test_metrics_on_hand_example() -> None:
    pairs = [
        ("поправка", "поправка"),
        ("поправка", "запрос"),
        ("запрос", "поправка"),
        ("запрос", "запрос"),
        ("запрос", "запрос"),
    ]
    m = compute_metrics(pairs, CATEGORIES)
    assert m.confusion[("поправка", "запрос")] == 1
    corr = m.classes["поправка"]
    assert (corr.precision, corr.recall, corr.support, corr.predicted) == (0.5, 0.5, 2, 2)
    req = m.classes["запрос"]
    assert req.precision == pytest.approx(2 / 3)
    assert req.recall == pytest.approx(2 / 3)


def test_empty_class_has_no_division_by_zero() -> None:
    m = compute_metrics([("запрос", "запрос")], CATEGORIES)
    empty = m.classes["инструкция"]
    assert (empty.precision, empty.recall, empty.support) == (None, None, 0)
    assert compute_metrics([], CATEGORIES).agreement is None


def test_agreement_and_deltas() -> None:
    assert compute_metrics([("а", "а"), ("а", "б")], CATEGORIES).agreement == 0.5
    cur = {"поправка": {"precision": 0.8, "recall": None}}
    prev = {"поправка": {"precision": 0.6, "recall": 0.5}}
    d = class_deltas(cur, prev)["поправка"]
    assert d["precision"] == pytest.approx(0.2)
    assert d["recall"] is None


def test_baseline() -> None:
    assert baseline_category("/commit", []) == "вызов скилла"
    assert baseline_category("нет, не так", []) == "поправка"
    assert baseline_category("сделай Y", ["отказ"]) == "поправка"
    assert baseline_category("всегда пиши тесты", []) == "инструкция"
    assert baseline_category("ок", []) == "нейтрально"
    assert baseline_category("добавь кнопку", []) == "запрос"


def test_stop_criteria() -> None:
    good = compute_metrics([("поправка", "поправка")] * 9 + [("запрос", "запрос")], CATEGORIES)
    bad = compute_metrics([("поправка", "запрос")] * 5 + [("запрос", "поправка")], CATEGORIES)
    checks = {c.name: c for c in stop_criteria(good, bad, 0.9)}
    assert checks["потолок"].ok is True
    assert checks["бейзлайн"].ok is True
    assert checks["ориентир"].ok is True
    assert stop_criteria(good, good, 0.5)[0].ok is False
    assert stop_criteria(good, good, None)[0].ok is None


def test_eval_default_is_dev_only(engine: Engine) -> None:
    _add(engine, "d1", "поправка", "поправка", text="нет, не так")
    _add(engine, "t1", "запрос", "поправка", split="test")
    dev = _run(engine)
    assert (dev.n, dev.split) == (1, "dev")
    test = _run(engine, split="test")
    assert (test.n, test.split) == (1, "test")


def test_eval_saves_run_and_second_run_shows_delta(engine: Engine) -> None:
    _add(engine, "d1", "поправка", "запрос")
    _add(engine, "d2", "запрос", "запрос")
    first = _run(engine)
    assert first.deltas is None
    with Session(engine) as s:
        s.add(
            ClassificationRow(
                episode_uuid="d1",
                category="поправка",
                gist="g",
                classifier_version=classifier_version("haiku"),
            )
        )
        s.commit()
    second = _run(engine)
    assert second.deltas is not None
    delta = second.deltas["поправка"]
    assert delta["recall"] == pytest.approx(1.0)
    with Session(engine) as s:
        runs = list(s.scalars(select(EvalRunRow).order_by(EvalRunRow.id)))
    assert len(runs) == 2
    assert runs[0].classifier_version == classifier_version("haiku")
    assert (runs[0].claude_version, runs[0].model, runs[0].split) == ("9.9", "haiku", "dev")
    assert runs[0].metrics["model"]["classes"]["поправка"]["recall"] == 0.0
    assert "baseline" in runs[0].metrics


def test_eval_lists_disagreements(engine: Engine) -> None:
    _add(engine, "d1", "поправка", "запрос", text="нет")
    result = _run(engine)
    assert [(d.uuid, d.human, d.model) for d in result.disagreements] == [
        ("d1", "поправка", "запрос")
    ]
    assert "d1" in render(result)


def test_eval_classifies_missing_with_model(engine: Engine) -> None:
    _add(engine, "d1", "запрос", None)
    calls: list[str] = []

    def call(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps({"category": "запрос", "gist": "g"})

    result = run_eval(engine, call, model="haiku", claude_version="1")
    assert len(calls) == 1
    assert calls[0].endswith("Реплика пользователя:\nреплика")
    assert result.disagreements == []


def test_ceiling_uses_same_metrics(engine: Engine) -> None:
    _add(engine, "d1", "поправка", "поправка")
    _add(engine, "d2", "запрос", "запрос")
    with Session(engine) as s:
        for uuid, cat in (("d1", "поправка"), ("d2", "инструкция")):
            s.add(
                GoldLabelRow(
                    episode_uuid=uuid, category=cat, split="dev", labeled_at=T0, is_relabel=True
                )
            )
        s.commit()
    result = _run(engine)
    assert result.ceiling == 0.5
    assert result.ceiling_n == 2
    assert result.n == 2  # повторные метки не попадают в эталон


def test_empty_gold_set(engine: Engine) -> None:
    with pytest.raises(EmptyGoldSetError, match="label"):
        _run(engine)


def test_eval_command_empty_gold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setattr("ai_retro.cli.make_claude_call", lambda **_: _fail)
    monkeypatch.setattr("ai_retro.cli.claude_version", lambda: "9.9")
    result = CliRunner().invoke(app, ["eval"])
    assert result.exit_code == 1
    assert "label" in result.output


def test_eval_command_prints_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_retro.db.session import make_engine, migrate

    db = tmp_path / "x.db"
    engine = make_engine(db)
    migrate(engine)
    _add(engine, "d1", "поправка", "поправка")
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(db))
    monkeypatch.setattr("ai_retro.cli.make_claude_call", lambda **_: _fail)
    monkeypatch.setattr("ai_retro.cli.claude_version", lambda: "9.9")
    result = CliRunner().invoke(app, ["eval"])
    assert result.exit_code == 0, result.output
    assert "precision" in result.output
    assert "accuracy" not in result.output


def test_eval_command_empty_gold_without_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = CliRunner().invoke(app, ["eval"])
    assert result.exit_code == 1
    assert "label" in result.output


def test_eval_saves_partial_results_then_raises(engine: Engine) -> None:
    from ai_retro.errors import ModelCallError

    _add(engine, "d1", "запрос", None)
    _add(engine, "d2", "запрос", None)

    def call(prompt: str) -> str:
        if "d2-текст" in prompt:
            raise ModelCallError("упало")
        return json.dumps({"category": "запрос", "gist": "g"})

    with Session(engine) as s:
        s.get(EpisodeRow, "d2").text = "d2-текст"  # type: ignore[union-attr]
        s.commit()
    with pytest.raises(ModelCallError):
        run_eval(engine, call, model="haiku", claude_version="1")
    with Session(engine) as s:
        saved = list(s.scalars(select(ClassificationRow.episode_uuid)))
    assert saved == ["d1"]
