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
