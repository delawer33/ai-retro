"""Область беседы: правила по пути, для scratch — модель по первой реплике."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_retro.classify.service import LOOP_MARKER
from ai_retro.errors import InvalidDomainRulesError, ModelCallError
from ai_retro.llm import ModelCall
from ai_retro.parsing.schemas import Conversation

DOMAINS = ("работа", "личные проекты", "учёба", "идеи", "философия", "жизнь", "иное")
OTHER = "иное"
# Беседы desktop-приложения без папки живут в каталоге с этим фрагментом пути.
SCRATCH_MARKER = "Claude/scratch-workspaces/"
FIRST_TEXT_LIMIT = 2000

# Страховка от петли (ADR 0002): тот же маркер, что у классификатора.
SYSTEM_PROMPT = f"""{LOOP_MARKER}
Ты определяешь область беседы пользователя с агентом по её первой реплике.
Области: работа; личные проекты; учёба; идеи; философия; жизнь; иное.
Выбери одну; если не подходит ни одна, ответь «иное».
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {"domain": {"type": "string", "enum": list(DOMAINS)}},
    "required": ["domain"],
    "additionalProperties": False,
}


def parse_rules(raw: str, *, home: Path) -> list[tuple[Path, str]]:
    rules: list[tuple[Path, str]] = []
    for part in raw.split(";"):
        if not part.strip():
            continue
        prefix, sep, domain = part.rpartition("=")
        prefix, domain = prefix.strip(), domain.strip()
        if not sep or not prefix or domain not in DOMAINS:
            raise InvalidDomainRulesError(part.strip())
        rules.append((home / prefix, domain))
    return rules


def is_scratch(project: str) -> bool:
    return SCRATCH_MARKER in f"{project.replace(chr(92), '/')}/"


def domain_by_rules(project: str, rules: list[tuple[Path, str]]) -> str | None:
    """Самый длинный подходящий префикс (по границе компонента пути) или None."""
    path = Path(project)
    matches = [(prefix, domain) for prefix, domain in rules if path.is_relative_to(prefix)]
    if not matches:
        return None
    return max(matches, key=lambda m: len(m[0].parts))[1]


def domain_by_model(call: ModelCall, first_text: str) -> str:
    raw = call(first_text[:FIRST_TEXT_LIMIT])
    try:
        domain = json.loads(raw)["domain"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ModelCallError(f"Модель вернула область не по схеме: {raw[:100]!r}") from exc
    if domain not in DOMAINS:
        raise ModelCallError(f"Модель вернула неизвестную область: {domain!r}")
    return str(domain)


def _domain_or_other(call: ModelCall, first_text: str) -> str:
    """Сбой модели на одной беседе не валит скан: она получает «иное» и будет пересмотрена."""
    try:
        return domain_by_model(call, first_text)
    except ModelCallError:
        return OTHER


def _first_text(conversation: Conversation) -> str | None:
    return next((r.text for r in conversation.records if r.text), None)


def assign_domains(
    conversations: list[Conversation],
    rules: list[tuple[Path, str]],
    call: ModelCall | None,
    *,
    threads: int,
) -> dict[str, str]:
    """conversation_id → область. Модель зовётся только для scratch, раз на беседу."""
    result: dict[str, str] = {}
    scratch: list[tuple[str, str]] = []
    for conv in conversations:
        project = conv.records[0].project
        if is_scratch(project):
            text = _first_text(conv)
            if call is not None and text is not None:
                scratch.append((conv.id, text))
                continue
            result[conv.id] = OTHER
        else:
            result[conv.id] = domain_by_rules(project, rules) or OTHER
    if scratch and call is not None:
        model_call = call
        with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
            answers = pool.map(lambda item: _domain_or_other(model_call, item[1]), scratch)
            result.update({cid: d for (cid, _), d in zip(scratch, answers, strict=True)})
    return result
