import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ai_retro.errors import AiRetroError, ModelCallError
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
        raise ModelCallError(f"Классификатор вернул ответ не по схеме: {raw[:100]!r}") from exc
    if category not in CATEGORIES or not isinstance(gist, str):
        raise ModelCallError(f"Классификатор вернул неизвестную категорию: {category!r}")
    return Classification(category=category, gist=gist)


@dataclass
class BatchOutcome:
    results: dict[str, Classification] = field(default_factory=dict)
    error: AiRetroError | None = None


def classify_many(
    call: ModelCall, items: Sequence[tuple[str, str]], *, threads: int
) -> BatchOutcome:
    """items: (uuid, prompt). Первая ошибка останавливает запуск новых вызовов, но уже
    полученные результаты возвращаются: вызывающий сохраняет их и только потом падает."""
    outcome = BatchOutcome()
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futures = {pool.submit(classify_episode, call, prompt): uuid for uuid, prompt in items}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                outcome.results[futures[future]] = future.result()
            except AiRetroError as exc:
                if outcome.error is None:
                    outcome.error = exc
                    for pending in futures:
                        pending.cancel()
    return outcome
