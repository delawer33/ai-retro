"""Бейзлайн: регулярка по словам-маркерам поправок + признаки. Без модели."""

import re

from ai_retro.parsing.episodes import NEUTRAL, SIGNAL_SKILL, skill_arguments

_CORRECTION = re.compile(
    r"\b(нет,|не так|не то|неправильно|неверно|ошибк\w*|я же (просил|говорил|писал)|"
    r"опять|снова|зачем ты|почему ты|верни|откати|сломал\w*|не надо|не нужно)",
    re.IGNORECASE,
)
_INSTRUCTION = re.compile(
    r"\b(всегда|никогда|впредь|запомни|в будущем|по умолчанию|предпочитаю|я использую)\b",
    re.IGNORECASE,
)
_NEUTRAL = re.compile(
    r"^\s*(ок|окей|ok|да|ага|спасибо|хорошо|супер|понял|ясно|отлично|продолжай|давай)\W*$",
    re.IGNORECASE,
)
_CORRECTION_SIGNALS = {"прерывание", "отказ"}


def baseline_category(text: str, signals: list[str]) -> str:
    if SIGNAL_SKILL in signals:
        text = skill_arguments(text)  # команда — признак, оценивается реплика-аргумент
        if not text:
            return NEUTRAL
    if _CORRECTION_SIGNALS & set(signals) or _CORRECTION.search(text):
        return "поправка"
    if _INSTRUCTION.search(text):
        return "инструкция"
    if _NEUTRAL.match(text):
        return "нейтрально"
    return "запрос"
