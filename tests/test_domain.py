import json
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from ai_retro.config import Settings
from ai_retro.db.models import EpisodeRow
from ai_retro.domain.service import (
    domain_by_model,
    domain_by_rules,
    is_scratch,
    parse_rules,
)
from ai_retro.errors import InvalidDomainRulesError, ModelCallError, UnrecognizedModelError
from ai_retro.scan.service import scan
from tests.conftest import NO_SKILLS, user_line

HOME = Path("/home/u")
RAW = "all/coll=учёба;all/projects=личные проекты;Documents/avito=работа"


def _log(logs: Path, name: str, cwd: str, text: str) -> None:
    d = logs / name
    d.mkdir(parents=True, exist_ok=True)
    line = user_line("u-" + name, text, "2026-09-25T11:00:00.000Z", sessionId="s-" + name, cwd=cwd)
    (d / "s.jsonl").write_text(line + "\n", encoding="utf-8")


class DomainModel:
    def __init__(self, answer: str = "философия") -> None:
        self.calls: list[str] = []
        self.answer = answer

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return json.dumps({"domain": self.answer})


def _scan(engine: Engine, logs: Path, model: DomainModel) -> None:
    scan(
        engine,
        lambda _: json.dumps({"category": "запрос", "gist": "g"}),
        logs_dir=logs,
        since=None,
        threads=2,
        classifier_version="v",
        domain_rules=parse_rules(RAW, home=HOME),
        domain_call=model,
        skills_dir=NO_SKILLS,
    )


def _domains(engine: Engine) -> dict[str, str | None]:
    with Session(engine) as s:
        return {r.conversation_id: r.domain for r in s.scalars(select(EpisodeRow))}


def test_rules_longest_prefix_and_boundary() -> None:
    rules = parse_rules(RAW + ";all/projects/x=идеи", home=HOME)
    assert domain_by_rules("/home/u/all/coll/!4c/js", rules) == "учёба"
    assert domain_by_rules("/home/u/all/projects/x/sub", rules) == "идеи"
    assert domain_by_rules("/home/u/all/projects", rules) == "личные проекты"
    assert domain_by_rules("/home/u/all/projects-old", rules) is None
    assert domain_by_rules("/home/u/Documents/sber", rules) is None


def test_bad_rules() -> None:
    with pytest.raises(InvalidDomainRulesError):
        parse_rules("all/coll=погода", home=HOME)
    with pytest.raises(InvalidDomainRulesError):
        parse_rules("all/coll", home=HOME)


def test_is_scratch() -> None:
    assert is_scratch("/home/u/.config/Claude/scratch-workspaces/a/b/scratch-2026-09-05")
    assert not is_scratch("/home/u/all/projects/x")


def test_domain_by_model_rejects_bad_answer() -> None:
    with pytest.raises(ModelCallError):
        domain_by_model(lambda _: json.dumps({"domain": "погода"}), "привет")


def test_scan_assigns_domains(engine: Engine, tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(logs, "a", "/home/u/all/coll/js", "лекция")
    _log(logs, "b", "/home/u/.config/Claude/scratch-workspaces/w1/w2/scratch-1", "о смысле")
    _log(logs, "c", "/tmp/elsewhere", "что-то")
    model = DomainModel()
    _scan(engine, logs, model)
    assert _domains(engine) == {"s-a": "учёба", "s-b": "философия", "s-c": "иное"}
    assert model.calls == ["о смысле"]


def test_rescan_keeps_domains(engine: Engine, tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(logs, "b", "/home/u/.config/Claude/scratch-workspaces/w1/w2/scratch-1", "о смысле")
    model = DomainModel()
    _scan(engine, logs, model)
    _scan(engine, logs, DomainModel("жизнь"))
    assert _domains(engine) == {"s-b": "философия"}
    assert len(model.calls) == 1


def test_failed_model_answer_does_not_abort_scan_and_is_retried(
    engine: Engine, tmp_path: Path
) -> None:
    logs = tmp_path / "logs"
    scratch = "/home/u/.config/Claude/scratch-workspaces/w1/w2/scratch-1"
    _log(logs, "a", "/home/u/all/coll/js", "лекция")
    _log(logs, "b", scratch, "о смысле")
    _scan(engine, logs, DomainModel("погода"))
    assert _domains(engine) == {"s-a": "учёба", "s-b": "иное"}
    _scan(engine, logs, DomainModel("философия"))
    assert _domains(engine) == {"s-a": "учёба", "s-b": "философия"}


SCRATCH = "/home/u/.config/Claude/scratch-workspaces/w1/w2/scratch-1"


def test_first_text_is_cleaned_user_reply_not_system_reminder(
    engine: Engine, tmp_path: Path
) -> None:
    logs = tmp_path / "logs"
    d = logs / "p"
    d.mkdir(parents=True)
    reminder = "<system-reminder>The user started this session without a folder. " + "x" * 2300
    lines = [
        user_line("r1", reminder + "</system-reminder>", "2026-09-25T11:00:00.000Z", cwd=SCRATCH),
        user_line("r2", "о смысле", "2026-09-25T11:00:05.000Z", parent="r1", cwd=SCRATCH),
    ]
    (d / "s.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    model = DomainModel()
    _scan(engine, logs, model)
    assert model.calls == ["о смысле"]
    assert _domains(engine) == {"s1": "философия"}


def test_legit_other_from_model_is_not_asked_again(engine: Engine, tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(logs, "b", SCRATCH, "о смысле")
    first = DomainModel("иное")
    _scan(engine, logs, first)
    second = DomainModel("жизнь")
    _scan(engine, logs, second)
    assert _domains(engine) == {"s-b": "иное"}
    assert first.calls == ["о смысле"]
    assert second.calls == []


def test_failure_is_logged(
    engine: Engine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    logs = tmp_path / "logs"
    _log(logs, "b", SCRATCH, "о смысле")
    _scan(engine, logs, DomainModel("погода"))
    assert "Область беседы не определена" in caplog.text


def test_unrecognized_model_aborts_scan_before_writing(engine: Engine, tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(logs, "a", "/home/u/all/coll/js", "лекция")
    _log(logs, "b", SCRATCH, "о смысле")

    def bad(_: str) -> str:
        raise UnrecognizedModelError("несуществующая")

    with pytest.raises(UnrecognizedModelError):
        scan(
            engine,
            lambda _: json.dumps({"category": "запрос", "gist": "g"}),
            logs_dir=logs,
            since=None,
            threads=2,
            classifier_version="v",
            domain_rules=parse_rules(RAW, home=HOME),
            domain_call=bad,
            skills_dir=NO_SKILLS,
        )
    assert _domains(engine) == {}


def test_announce_precedes_first_domain_call_and_counts_it(engine: Engine, tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(logs, "a", "/home/u/all/coll/js", "лекция")
    _log(logs, "b", SCRATCH, "о смысле")
    events: list[str] = []

    def domain_call(_: str) -> str:
        events.append("domain")
        return json.dumps({"domain": "жизнь"})

    def call(_: str) -> str:
        events.append("call")
        return json.dumps({"category": "запрос", "gist": "g"})

    scan(
        engine,
        call,
        logs_dir=logs,
        since=None,
        threads=1,
        classifier_version="v",
        announce=lambda n: events.append(f"announce {n}"),
        domain_rules=parse_rules(RAW, home=HOME),
        domain_call=domain_call,
        skills_dir=NO_SKILLS,
    )
    assert events[0] == "announce 3"  # 2 эпизода + 1 вызов области
    assert events.count("domain") == 1


def test_tilde_in_rule_prefix_expands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/u")
    monkeypatch.setenv("AI_RETRO_DOMAIN_RULES", "~/notes=идеи;all/coll=учёба")
    rules = parse_rules(Settings().domain_rules, home=Path("/elsewhere"))
    assert rules == [(Path("/home/u/notes"), "идеи"), (Path("/elsewhere/all/coll"), "учёба")]
