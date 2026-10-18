"""Текстовый вывод результата eval: матрица, метрики по классам, дельта, стоп-критерий."""

from ai_retro.classify.service import CATEGORIES
from ai_retro.evals.metrics import Metrics
from ai_retro.evals.service import EvalResult

_SNIPPET = 80


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _delta(value: float | None) -> str:
    return "" if value is None else f" ({value:+.2f})"


def _confusion(metrics: Metrics) -> list[str]:
    corner = "человек \\ модель"
    width = max(len(c) for c in CATEGORIES) + 2
    first = max(len(corner), width) + 2
    lines = [corner.ljust(first) + "".join(c.ljust(width) for c in CATEGORIES)]
    for human in CATEGORIES:
        row = (str(metrics.confusion.get((human, model), 0)) for model in CATEGORIES)
        lines.append(human.ljust(first) + "".join(v.ljust(width) for v in row))
    return lines


def _per_class(metrics: Metrics, deltas: dict[str, dict[str, float | None]] | None) -> list[str]:
    lines = []
    for cat in CATEGORIES:
        c = metrics.classes[cat]
        d = deltas[cat] if deltas else {}
        lines.append(
            f"{cat:<14} precision {_num(c.precision)}{_delta(d.get('precision'))}  "
            f"recall {_num(c.recall)}{_delta(d.get('recall'))}  "
            f"(в эталоне {c.support}, предсказано {c.predicted})"
        )
    return lines


def render(result: EvalResult) -> str:
    lines = [f"Эталон: {result.split}, эпизодов {result.n}", "", "Матрица ошибок"]
    lines += _confusion(result.model)
    lines += ["", "Модель" + (" (в скобках — к прошлому прогону)" if result.deltas else "")]
    lines += _per_class(result.model, result.deltas)
    lines += ["", "Бейзлайн (регулярка + признаки)"]
    lines += _per_class(result.baseline, None)
    ceiling = "нет повторных меток" if result.ceiling is None else f"{result.ceiling:.2f}"
    lines += ["", f"Потолок (согласие с собой, пар {result.ceiling_n}): {ceiling}"]
    lines += ["", "Стоп-критерий этапа 1"]
    for check in result.checks:
        mark = {True: "ok", False: "НЕТ", None: "?"}[check.ok]
        lines.append(f"[{mark}] {check.name}: {check.detail}")
    lines.append("Проверка по паттернам (> 2 фальшивых из топ-10) — глазами, по отчёту.")
    lines += ["", f"Расхождения с человеком: {len(result.disagreements)}"]
    for d in result.disagreements:
        snippet = d.text.replace("\n", " ")[:_SNIPPET]
        lines.append(f"{d.uuid}  человек: {d.human}, модель: {d.model}  «{snippet}»")
    return "\n".join(lines)
