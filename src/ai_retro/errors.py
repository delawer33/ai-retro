class AiRetroError(Exception):
    """Ожидаемая ошибка: сообщение говорит пользователю, что делать."""


class LogsDirNotFoundError(AiRetroError):
    def __init__(self, path: object) -> None:
        super().__init__(
            f"Каталог логов не найден: {path}. Задай AI_RETRO_LOGS_DIR или проверь путь."
        )


class ClaudeNotFoundError(AiRetroError):
    def __init__(self) -> None:
        super().__init__("Команда `claude` не найдена в PATH. Установи Claude Code и войди в него.")


class ModelCallError(AiRetroError):
    pass


class InvalidSinceError(AiRetroError):
    def __init__(self, value: str) -> None:
        super().__init__(
            f"Не понимаю --since {value!r}: ожидается число и единица, напр. 12h, 1d, 2w."
        )


class LabelExistsError(AiRetroError):
    def __init__(self, uuid: str) -> None:
        super().__init__(
            f"Эпизод {uuid} уже размечен: эталон не переписывается. "
            "Для повторной разметки используй --relabel."
        )


class NothingToLabelError(AiRetroError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)


class InvalidDomainRulesError(AiRetroError):
    def __init__(self, part: str) -> None:
        super().__init__(
            f"Не понимаю правило области {part!r} в AI_RETRO_DOMAIN_RULES: "
            "ожидается «префикс=область», правила через «;», область из списка в CONTEXT.md."
        )


class NoClassificationsError(AiRetroError):
    def __init__(self) -> None:
        super().__init__("Нет классификаций за период. Запусти `ai-retro scan`, затем повтори.")


class InvalidReportPathError(AiRetroError):
    def __init__(self, path: object) -> None:
        super().__init__(
            f"Отчёт нельзя писать в {path}: отчёт не правит CLAUDE.md и SKILL.md. "
            "Укажи другой --output."
        )


class ReportWriteError(AiRetroError):
    def __init__(self, path: object, reason: str) -> None:
        super().__init__(f"Отчёт нельзя записать в {path}: {reason}. Укажи другой --output.")


class EmptyGoldSetError(AiRetroError):
    def __init__(self, split: str) -> None:
        super().__init__(f"В эталоне нет эпизодов ({split}). Разметь их командой `ai-retro label`.")
