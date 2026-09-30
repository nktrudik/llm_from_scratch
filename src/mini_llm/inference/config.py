"""Конфигурация генерации ответа модели."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mini_llm.tokenization.config import DEFAULT_TOKENIZER_PATH


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Пути и параметры одного запуска генерации."""

    checkpoint_file: Path = Path("checkpoints/training/best.pt")
    tokenizer_file: Path = DEFAULT_TOKENIZER_PATH
    device: str = "cuda"
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int | None = 50

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens не может быть отрицательным")
        if self.temperature <= 0.0:
            raise ValueError("temperature должна быть положительной")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k должен быть положительным")
