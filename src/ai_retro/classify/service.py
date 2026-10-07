import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_retro.db.models import ClassificationRow
from ai_retro.errors import AiRetroError, ClassifierSchemaError
from ai_retro.llm import ModelCall

CATEGORIES = ("поправка", "инструкция", "запрос", "нейтрально")
# Страховка от петли (ADR 0002): scan пропускает беседы, где встречается маркер.
LOOP_MARKER = "[ai-retro-classifier]"
# Гайд по разметке — одновременно системный промпт классификатора.
GUIDE_PATH = Path(__file__).with_name("guide.md")
SYSTEM_PROMPT = GUIDE_PATH.read_text(encoding="utf-8")

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "gist": {"type": "string"},
    },
    "required": ["category", "gist"],
    "additionalProperties": False,
}


def prompt_sha(prompt: str = SYSTEM_PROMPT) -> str:
    """Git-sha блоба промпта (как `git hash-object guide.md`), без вызова git."""
    body = prompt.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(body) + body, usedforsecurity=False).hexdigest()


def classifier_version(model: str, prompt: str = SYSTEM_PROMPT) -> str:
    return f"{prompt_sha(prompt)[:12]}+{model}"


@dataclass(frozen=True)
class EpisodeInput:
    """Всё, что классификатор видит об эпизоде."""

    text: str
    signals: Sequence[str] = ()
    agent_reply: str | None = None
    abandoned_branch: str | None = None


def _tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else "…" + text[-limit:]


def _head(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def build_prompt(episode: EpisodeInput, *, context_chars: int) -> str:
    """Контекст эпизода: признаки, ответ агента (хвост), брошенная ветка (начало), реплика."""
    parts = [f"Признаки: {', '.join(episode.signals) if episode.signals else 'нет'}"]
    if episode.agent_reply:
        parts.append(
            "Последний ответ агента перед репликой:\n" + _tail(episode.agent_reply, context_chars)
        )
    if "rewind" in episode.signals and episode.abandoned_branch:
        parts.append("Начало брошенной ветки:\n" + _head(episode.abandoned_branch, context_chars))
    parts.append("Реплика пользователя:\n" + episode.text)
    return "\n\n".join(parts)


@dataclass(frozen=True)
class Classification:
    category: str
    gist: str


def classify_episode(call: ModelCall, prompt: str) -> Classification:
    raw = call(prompt)
    try:
        data = json.loads(raw)
        category, gist = data["category"], data["gist"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ClassifierSchemaError(
            f"Классификатор вернул ответ не по схеме: {raw[:100]!r}"
        ) from exc
    if category not in CATEGORIES or not isinstance(gist, str):
        raise ClassifierSchemaError(f"Классификатор вернул неизвестную категорию: {category!r}")
    return Classification(category=category, gist=gist)


def classify_with_retry(call: ModelCall, prompt: str) -> Classification:
    """Ответ не по схеме — один повтор; второй раз не по схеме — ошибка уходит наверх."""
    try:
        return classify_episode(call, prompt)
    except ClassifierSchemaError:
        return classify_episode(call, prompt)


@dataclass
class BatchOutcome:
    results: dict[str, Classification] = field(default_factory=dict)
    # Ответ не по схеме и на повторе (ADR 0010): эпизод остаётся без классификации.
    unclassified: list[str] = field(default_factory=list)
    error: AiRetroError | None = None


def classify_many(
    call: ModelCall, items: Sequence[tuple[str, str]], *, threads: int
) -> BatchOutcome:
    """items: (uuid, prompt). Ответ не по схеме после повтора пропускается (unclassified).
    Первая другая ошибка (вызов невозможен) останавливает запуск новых вызовов, но уже
    полученные результаты возвращаются: вызывающий сохраняет их и только потом падает."""
    outcome = BatchOutcome()
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futures = {pool.submit(classify_with_retry, call, prompt): uuid for uuid, prompt in items}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                outcome.results[futures[future]] = future.result()
            except ClassifierSchemaError:
                outcome.unclassified.append(futures[future])
            except AiRetroError as exc:
                if outcome.error is None:
                    outcome.error = exc
                    for pending in futures:
                        pending.cancel()
    return outcome


def latest_classifications(
    session: Session, *, classifier_version: str | None = None
) -> dict[str, ClassificationRow]:
    """Актуальная классификация каждого эпизода — последняя вставленная (ADR 0010).

    Без `classifier_version` — среди всех версий; с ним — среди строк этой версии.
    """
    stmt = select(ClassificationRow).order_by(ClassificationRow.id)
    if classifier_version is not None:
        stmt = stmt.where(ClassificationRow.classifier_version == classifier_version)
    return {row.episode_uuid: row for row in session.scalars(stmt)}
