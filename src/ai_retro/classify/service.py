import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ai_retro.errors import ModelCallError
from ai_retro.llm import ModelCall

CLASSIFIER_VERSION = "v0"
CATEGORIES = ("поправка", "инструкция", "запрос", "вызов скилла", "нейтрально")
# Страховка от петли (ADR 0002): scan пропускает беседы, где встречается маркер.
LOOP_MARKER = "[ai-retro-classifier]"

SYSTEM_PROMPT = f"""{LOOP_MARKER}
Ты классифицируешь одну реплику пользователя из беседы с агентом.
Категория (одна; если подходят несколько, старшая: поправка > инструкция > запрос > нейтрально):
- поправка: реплика реагирует на то, что агент сделал не так, включая «я же просил»;
- инструкция: правило, факт или предпочтение, которые переживут текущую задачу;
- запрос: новая задача или продолжение текущей;
- вызов скилла: вызов слэш-команды;
- нейтрально: ничего из перечисленного.
Суть: одна фраза о том, что именно пользователь поправил или сообщил, без привязки к словам реплики.
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "gist": {"type": "string"},
    },
    "required": ["category", "gist"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Classification:
    category: str
    gist: str


def classify_episode(call: ModelCall, text: str) -> Classification:
    raw = call(text)
    try:
        data = json.loads(raw)
        category, gist = data["category"], data["gist"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ModelCallError(f"Классификатор вернул ответ не по схеме: {raw[:100]!r}") from exc
    if category not in CATEGORIES or not isinstance(gist, str):
        raise ModelCallError(f"Классификатор вернул неизвестную категорию: {category!r}")
    return Classification(category=category, gist=gist)


def classify_many(
    call: ModelCall, items: Sequence[tuple[str, str]], *, threads: int
) -> dict[str, Classification]:
    """items: (uuid, text). Ошибка модели на одном эпизоде прерывает прогон."""
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        results = pool.map(lambda item: classify_episode(call, item[1]), items)
        return {uuid: result for (uuid, _), result in zip(items, results, strict=True)}
