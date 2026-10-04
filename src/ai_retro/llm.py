"""Единственное место, где вызывается `claude` (ADR 0002)."""

import json
import shutil
import subprocess
from collections.abc import Callable
from typing import Any

from ai_retro.errors import ClaudeNotFoundError, ModelCallError

# «промпт → ответ» (JSON-строка по схеме); передаётся сервисам явно.
ModelCall = Callable[[str], str]

TIMEOUT_SECONDS = 180


def claude_args(*, system_prompt: str, json_schema: dict[str, Any], model: str) -> list[str]:
    return [
        "claude",
        "-p",
        "--safe-mode",
        "--system-prompt",
        system_prompt,
        "--tools",
        "",
        "--model",
        model,
        "--json-schema",
        json.dumps(json_schema),
        "--output-format",
        "json",
        "--no-session-persistence",
    ]


def make_claude_call(*, system_prompt: str, json_schema: dict[str, Any], model: str) -> ModelCall:
    if shutil.which("claude") is None:
        raise ClaudeNotFoundError
    args = claude_args(system_prompt=system_prompt, json_schema=json_schema, model=model)

    def call(prompt: str) -> str:
        try:
            proc = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ModelCallError("Вызов `claude` завис: превышено время ожидания.") from exc
        if proc.returncode != 0:
            raise ModelCallError(
                f"`claude` завершился с кодом {proc.returncode}: {proc.stderr.strip()[:200]}"
            )
        return _extract_payload(proc.stdout)

    return call


def _extract_payload(stdout: str) -> str:
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ModelCallError("`claude` вернул не JSON.") from exc
    if isinstance(envelope, dict):
        if envelope.get("is_error"):
            raise ModelCallError(f"`claude` вернул ошибку: {str(envelope.get('result'))[:200]}")
        if envelope.get("structured_output") is not None:
            return json.dumps(envelope["structured_output"])
        if isinstance(envelope.get("result"), str):
            return str(envelope["result"])
    raise ModelCallError("В ответе `claude` нет структурированного результата.")
