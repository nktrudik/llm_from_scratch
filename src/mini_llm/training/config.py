"""Конфигурация полного цикла обучения модели."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mini_llm.bpe_tokenizer import DEFAULT_TOKENIZER_PATH
from mini_llm.config import MAX_TRAINING_BATCH_SIZE
from mini_llm.training.progress import calculate_training_plan


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Параметры однопроцессного запуска обучения."""

    splits_dir: Path = Path("data/processed/splits")
    tokenizer_file: Path = DEFAULT_TOKENIZER_PATH
    token_statistics_file: Path = Path("data/processed/token_statistics.json")
    checkpoint_dir: Path = Path("checkpoints/training")
    resume_from: Path | None = None
    batch_size: int = MAX_TRAINING_BATCH_SIZE
    num_workers: int = 0
    epochs: int | None = 3
    max_steps: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    validation_interval: int = 1000
    checkpoint_interval: int = 1000
    validation_batches: int = 200
    log_interval: int = 10
    random_seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= MAX_TRAINING_BATCH_SIZE:
            raise ValueError(
                f"batch_size должен быть в диапазоне от 1 до {MAX_TRAINING_BATCH_SIZE}"
            )
        positive_values = {
            "validation_interval": self.validation_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "log_interval": self.log_interval,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} должен быть положительным")
        calculate_training_plan(
            1,
            self.batch_size,
            epochs=self.epochs,
            max_steps=self.max_steps,
        )
        if self.num_workers < 0 or self.validation_batches < 0:
            raise ValueError("num_workers и validation_batches не могут быть отрицательными")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate должен быть положительным")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay не может быть отрицательным")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm должен быть положительным")
