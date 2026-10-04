"""Стоп-критерий этапа 1: проверки 1–3 (четвёртая, по паттернам, — глазами)."""

from dataclasses import dataclass

from ai_retro.evals.metrics import Metrics

CEILING_MIN = 0.75
BASELINE_GAP = 0.15
CORRECTION = "поправка"
PRECISION_TARGET = 0.8
RECALL_TARGET = 0.6


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool | None  # None: не хватает данных
    detail: str


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def stop_criteria(model: Metrics, baseline: Metrics, ceiling: float | None) -> list[Check]:
    m, b = model.classes[CORRECTION], baseline.classes[CORRECTION]
    checks = []

    if ceiling is None:
        checks.append(
            Check(
                "потолок", None, "нет повторной разметки: ai-retro label, режим повторной разметки"
            )
        )
    else:
        checks.append(
            Check(
                "потолок",
                ceiling >= CEILING_MIN,
                f"согласие с собой {ceiling:.2f} (нужно ≥ {CEILING_MIN}); иначе чинить категории",
            )
        )

    if m.precision is None or b.precision is None:
        checks.append(Check("бейзлайн", None, "precision поправок не определена"))
    else:
        gap = m.precision - b.precision
        checks.append(
            Check(
                "бейзлайн",
                gap >= BASELINE_GAP,
                f"precision поправок: LLM {m.precision:.2f}, регулярка {b.precision:.2f}, "
                f"разница {gap * 100:+.0f} п.п. (нужно ≥ {BASELINE_GAP * 100:.0f}); "
                "иначе упрощение: категории ставит регулярка",
            )
        )

    if m.precision is None or m.recall is None:
        checks.append(Check("ориентир", None, "precision/recall поправок не определены"))
    else:
        target = PRECISION_TARGET if ceiling is None else min(PRECISION_TARGET, ceiling)
        checks.append(
            Check(
                "ориентир",
                m.precision >= target and m.recall >= RECALL_TARGET,
                f"precision поправок {m.precision:.2f} (нужно ≥ {_fmt(target)}, не выше потолка), "
                f"recall {m.recall:.2f} (нужно ≥ {RECALL_TARGET})",
            )
        )
    return checks
