"""Конфигурация генерации ответа модели."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mini_llm.pretrained.workspace import DEFAULT_MODEL_ID, ModelPaths
from mini_llm.tokenization.config import DEFAULT_TOKENIZER_PATH

DEFAULT_CUSTOM_CHECKPOINT_PATH = Path("checkpoints/training/best.pt")
DEFAULT_PRETRAINED_CHECKPOINT_PATH = (
    ModelPaths.for_model(DEFAULT_MODEL_ID).checkpoint_dir / "best.pt"
)
DEFAULT_PRETRAINED_CONFIG_PATH = ModelPaths.for_model(DEFAULT_MODEL_ID).config_file
PretrainedGenerationMode = Literal["before_sft", "after_sft", "custom_checkpoint"]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Пути и параметры одного запуска генерации."""

    checkpoint_file: Path | None = DEFAULT_CUSTOM_CHECKPOINT_PATH
    tokenizer_file: Path | None = DEFAULT_TOKENIZER_PATH
    device: str = "cuda"
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_k: int | None = 50
    model_backend: Literal["custom", "pretrained"] = "custom"
    pretrained_config_file: Path | None = None
    pretrained_mode: PretrainedGenerationMode = "after_sft"

    def __post_init__(self) -> None:
        if self.model_backend == "custom" and self.tokenizer_file is None:
            raise ValueError("Для custom backend нужен tokenizer_file")
        if self.model_backend == "custom" and self.checkpoint_file is None:
            raise ValueError("Для custom backend нужен checkpoint_file")
        if self.model_backend == "pretrained" and self.pretrained_config_file is None:
            raise ValueError("Для pretrained backend нужен pretrained_config_file")
        if self.model_backend == "pretrained":
            if (
                self.pretrained_mode in {"after_sft", "custom_checkpoint"}
                and self.checkpoint_file is None
            ):
                raise ValueError(f"Для режима {self.pretrained_mode} нужен checkpoint_file")
            if self.pretrained_mode == "before_sft" and self.checkpoint_file is not None:
                raise ValueError("Режим before_sft не использует checkpoint_file")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens не может быть отрицательным")
        if self.temperature <= 0.0:
            raise ValueError("temperature должна быть положительной")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k должен быть положительным")
