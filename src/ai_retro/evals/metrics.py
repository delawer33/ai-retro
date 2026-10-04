"""Чистые метрики качества: матрица ошибок, precision/recall по категориям, согласие, дельта."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# (эталон/человек, предсказание/модель)
Pair = tuple[str, str]


@dataclass(frozen=True)
class ClassMetrics:
    precision: float | None  # None: класс ни разу не предсказан
    recall: float | None  # None: класса нет в эталоне
    support: int  # сколько эпизодов этого класса в эталоне
    predicted: int  # сколько раз класс предсказан


@dataclass(frozen=True)
class Metrics:
    n: int
    confusion: dict[Pair, int]
    classes: dict[str, ClassMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "confusion": [[h, m, c] for (h, m), c in sorted(self.confusion.items())],
            "classes": {
                name: {
                    "precision": c.precision,
                    "recall": c.recall,
                    "support": c.support,
                    "predicted": c.predicted,
                }
                for name, c in self.classes.items()
            },
        }


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def compute_metrics(pairs: Sequence[Pair], categories: Sequence[str]) -> Metrics:
    confusion = Counter(pairs)
    classes: dict[str, ClassMetrics] = {}
    for cat in categories:
        true_pos = confusion[(cat, cat)]
        support = sum(c for (h, _), c in confusion.items() if h == cat)
        predicted = sum(c for (_, m), c in confusion.items() if m == cat)
        classes[cat] = ClassMetrics(
            precision=_ratio(true_pos, predicted),
            recall=_ratio(true_pos, support),
            support=support,
            predicted=predicted,
        )
    return Metrics(n=len(pairs), confusion=dict(confusion), classes=classes)


def agreement(pairs: Sequence[Pair]) -> float | None:
    """Доля совпавших пар; для потолка: первая и повторная метка человека."""
    return _ratio(sum(1 for a, b in pairs if a == b), len(pairs))


def class_deltas(
    current: dict[str, Any], previous: dict[str, Any]
) -> dict[str, dict[str, float | None]]:
    """Дельта precision/recall по классам между двумя сериализованными `Metrics.classes`."""
    result: dict[str, dict[str, float | None]] = {}
    for cat, cur in current.items():
        prev = previous.get(cat)
        result[cat] = {
            key: _diff(cur.get(key), prev.get(key) if prev else None)
            for key in ("precision", "recall")
        }
    return result


def _diff(cur: float | None, prev: float | None) -> float | None:
    return None if cur is None or prev is None else cur - prev
