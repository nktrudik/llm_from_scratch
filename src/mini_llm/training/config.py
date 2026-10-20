"""Конфигурация полного цикла обучения модели."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mini_llm.data.config import MAX_BATCH_SIZE
from mini_llm.tokenization.config import DEFAULT_TOKENIZER_PATH
from mini_llm.training.progress import calculate_training_plan

LEGACY_SFT_SETTINGS: dict[str, int | str | bool] = {
    "gradient_accumulation_steps": 1,
    "lr_scheduler_type": "constant",
    "warmup_steps": 0,
    "packing": False,
}


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Параметры однопроцессного запуска обучения."""

    splits_dir: Path = Path("data/datasets/2ch/v3/splits")
    tokenizer_file: Path = DEFAULT_TOKENIZER_PATH
    token_statistics_file: Path = Path("data/datasets/2ch/v3/reports/token_statistics.json")
    checkpoint_dir: Path = Path("checkpoints/unmanaged/custom")
    resume_from: Path | None = None
    model_backend: Literal["custom", "pretrained"] = "custom"
    pretrained_config_file: Path | None = None
    batch_size: int = MAX_BATCH_SIZE
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
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    pretrained_engine: Literal["transformers", "unsloth"] = "transformers"
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: Literal["constant", "cosine"] = "constant"
    warmup_steps: int = 0
    packing: bool = False

    def __post_init__(self) -> None:
        if self.model_backend == "pretrained" and self.pretrained_config_file is None:
            raise ValueError("Для pretrained backend нужен pretrained_config_file")
        if self.model_backend == "custom" and self.pretrained_config_file is not None:
            raise ValueError("pretrained_config_file допустим только для pretrained backend")
        if not 1 <= self.batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size должен быть в диапазоне от 1 до {MAX_BATCH_SIZE}")
        positive_values = {
            "validation_interval": self.validation_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "log_interval": self.log_interval,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
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
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps не может быть отрицательным")
        if self.lr_scheduler_type not in {"constant", "cosine"}:
            raise ValueError("lr_scheduler_type должен быть constant или cosine")
        if self.model_backend == "custom" and (
            self.gradient_accumulation_steps != 1
            or self.lr_scheduler_type != "constant"
            or self.warmup_steps != 0
            or self.packing
        ):
            raise ValueError("Accumulation, scheduler, warmup и packing доступны только для SFT")
        if self.max_train_samples is not None and self.max_train_samples <= 0:
            raise ValueError("max_train_samples должен быть положительным или None (весь train)")
        if self.max_validation_samples is not None and self.max_validation_samples <= 0:
            raise ValueError(
                "max_validation_samples должен быть положительным или None (весь validation)"
            )


@dataclass(frozen=True, slots=True)
class PretrainedTrainingConfig(TrainingConfig):
    """Стартовый профиль pretrained-обучения для GPU с 4 GB VRAM."""

    model_backend: Literal["custom", "pretrained"] = "pretrained"
    batch_size: int = 1
    gradient_accumulation_steps: int = 2
    lr_scheduler_type: Literal["constant", "cosine"] = "cosine"
    warmup_steps: int = 30
    packing: bool = True
    num_workers: int = 2
    max_train_samples: int | None = 30_000
    max_validation_samples: int | None = 1000
    log_interval: int = 100
    validation_interval: int = 50
    validation_batches: int = 50
    checkpoint_interval: int = 50
