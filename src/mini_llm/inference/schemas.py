"""Схемы результата генерации."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Текст и идентификаторы сгенерированных токенов."""

    text: str
    token_ids: list[int]
