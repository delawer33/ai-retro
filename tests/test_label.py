from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from ai_retro.cli import app
from ai_retro.db.models import ClassificationRow, EpisodeRow, GoldLabelRow
from ai_retro.errors import LabelExistsError
from ai_retro.label.service import (
    Candidate,
    LabelItem,
    draw_sample,
    ensure_sample,
    pending_items,
    pending_relabel_items,
    render_item,
    save_label,
)

NOW = datetime(2026, 10, 4, tzinfo=UTC)


def _candidates() -> list[Candidate]:
    result = []
    for i in range(400):
        kind = i % 4  # 0 признаки, 1-2 ненейтрально, 3 нейтрально
        result.append(
            Candidate(
                uuid=f"e{i}",
                conversation_id=f"c{i // 5}",
                project=f"p{i % 7}",
                has_signals=kind == 0,
                model_category="нейтрально" if kind == 3 else "запрос",
            )
        )
    return result


def test_sample_keeps_strata_and_conversation_limit() -> None:
    sample = draw_sample(_candidates())
    assert len(sample) == 100
    by_uuid = {c.uuid: c for c, _ in sample}
    strata = Counter(
        0 if c.has_signals else (2 if c.model_category == "нейтрально" else 1)
        for c in by_uuid.values()
    )
    assert strata == {0: 40, 1: 40, 2: 20}
    per_conversation = Counter(c.conversation_id for c in by_uuid.values())
    assert max(per_conversation.values()) <= 3
    assert len({c.project for c in by_uuid.values()}) > 1


def test_split_is_deterministic_and_about_30_percent_test() -> None:
    first = draw_sample(_candidates())
    second = draw_sample(list(reversed(_candidates())))
    assert {(c.uuid, s) for c, s in first} == {(c.uuid, s) for c, s in second}
    assert Counter(s for _, s in first) == {"dev": 70, "test": 30}


def test_sample_takes_what_exists_when_stratum_is_short() -> None:
    few = [
        Candidate(f"e{i}", f"c{i}", "p", has_signals=False, model_category="запрос")
        for i in range(10)
    ]
    assert len(draw_sample(few)) == 10


def test_unclassified_without_signals_is_not_sampled() -> None:
    c = Candidate("x", "c", "p", has_signals=False, model_category=None)
    assert draw_sample([c]) == []


def _fill(engine: Engine, n: int = 30) -> None:
    with Session(engine) as s:
        for i in range(n):
            s.add(
                EpisodeRow(
                    uuid=f"e{i}",
                    conversation_id=f"c{i}",
                    project=f"p{i % 3}",
                    timestamp=NOW,
                    text=f"реплика {i}",
                    context="ответ агента" if i == 0 else None,
                    signals=["rewind"] if i % 3 == 0 else [],
                )
            )
            s.add(
                ClassificationRow(
                    episode_uuid=f"e{i}",
                    category="поправка" if i % 2 else "нейтрально",
                    gist="СЕКРЕТНАЯ-СУТЬ",
                    classifier_version="v0",
                )
            )
        s.commit()


def test_resume_starts_from_first_unlabeled(engine: Engine) -> None:
    _fill(engine)
    assert ensure_sample(engine, size=10) == 10
    items = pending_items(engine)
    save_label(engine, items[0].uuid, "запрос", relabel=False, now=NOW)
    save_label(engine, items[1].uuid, "запрос", relabel=False, now=NOW)
    assert [i.uuid for i in pending_items(engine)] == [i.uuid for i in items[2:]]
    assert ensure_sample(engine, size=10) == 0  # выборка не пересоздаётся


def test_extend_adds_new_episodes(engine: Engine) -> None:
    _fill(engine)
    ensure_sample(engine, size=10)
    before = {i.uuid for i in pending_items(engine)}
    assert ensure_sample(engine, size=10, extend=True) > 0
    assert before < {i.uuid for i in pending_items(engine)}


def test_original_label_is_never_overwritten(engine: Engine) -> None:
    _fill(engine)
    ensure_sample(engine, size=10)
    uuid = pending_items(engine)[0].uuid
    save_label(engine, uuid, "запрос", relabel=False, now=NOW)
    with pytest.raises(LabelExistsError):
        save_label(engine, uuid, "поправка", relabel=False, now=NOW)
    save_label(engine, uuid, "поправка", relabel=True, now=NOW)
    with Session(engine) as s:
        rows = list(s.scalars(select(GoldLabelRow).order_by(GoldLabelRow.id)))
    assert [(r.category, r.is_relabel) for r in rows] == [("запрос", False), ("поправка", True)]
    assert rows[0].split == rows[1].split


def test_relabel_picks_labeled_episodes_only(engine: Engine) -> None:
    _fill(engine)
    ensure_sample(engine, size=10)
    items = pending_items(engine)
    assert pending_relabel_items(engine) == []
    for item in items[:4]:
        save_label(engine, item.uuid, "запрос", relabel=False, now=NOW)
    todo = pending_relabel_items(engine)
    assert {i.uuid for i in todo} == {i.uuid for i in items[:4]}
    save_label(engine, todo[0].uuid, "запрос", relabel=True, now=NOW)
    assert len(pending_relabel_items(engine)) == 3


def test_render_has_no_model_label() -> None:
    item = LabelItem("u", "моя реплика", "ответ агента", ("rewind",))
    out = render_item(item, number=1, total=5)
    assert "моя реплика" in out
    assert "ответ агента" in out
    assert "rewind" in out


@pytest.fixture
def cli_env(engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Engine:
    _fill(engine)
    monkeypatch.setenv("AI_RETRO_DB_PATH", str(tmp_path / "test.db"))
    return engine


def _labels(engine: Engine) -> list[GoldLabelRow]:
    with Session(engine) as s:
        return list(s.scalars(select(GoldLabelRow).order_by(GoldLabelRow.id)))


def test_cli_label_scripted_input_and_resume(cli_env: Engine) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["label"], input="1s3q")
    assert result.exit_code == 0, result.output
    assert "СЕКРЕТНАЯ-СУТЬ" not in result.output
    labels = _labels(cli_env)
    assert [x.category for x in labels] == ["поправка", "запрос"]
    # skipped episode comes first on resume
    result = runner.invoke(app, ["label"], input="5q")
    assert result.exit_code == 0, result.output
    assert len(_labels(cli_env)) == 3


def test_cli_relabel_blind(cli_env: Engine) -> None:
    runner = CliRunner()
    runner.invoke(app, ["label"], input="11")
    result = runner.invoke(app, ["label", "--relabel"], input="2q")
    assert result.exit_code == 0, result.output
    labels = _labels(cli_env)
    assert [(x.category, x.is_relabel) for x in labels] == [
        ("поправка", False),
        ("поправка", False),
        ("инструкция", True),
    ]
    assert "поправка" not in result.output.split("Реплика:")[1].split("1=")[0]
