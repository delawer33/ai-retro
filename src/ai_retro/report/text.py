"""Реплики пользователя в тексте отчёта и промптов: вставки скрыты, длина ограничена."""

import re

PASTE_MARKER = "[вставка]"
EXAMPLE_LIMIT = 200

_PASTE_RE = re.compile(
    r"<pasted_content>.*?</pasted_content>|```.*?```|\[(?:Pasted text|Image)[^\]]*\]",
    re.DOTALL,
)


def mask_pastes(text: str) -> str:
    """Вставки (<pasted_content>, блоки кода, плейсхолдеры) показываются маркером с длиной."""
    return _PASTE_RE.sub(lambda m: f"[вставка, {len(m.group(0))} симв.]", text)


def example_text(text: str, *, limit: int = EXAMPLE_LIMIT) -> str:
    flat = " ".join(mask_pastes(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
