"""Вспомогательные операции окружения и checkpoint-файлов."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import Optimizer

from mini_llm.modeling import CausalLMBackend
from mini_llm.training.checkpoints import save_checkpoint
from mini_llm.training.config import TrainingConfig
from mini_llm.training.schemas import TrainingState


def select_device(name: str) -> torch.device:
    """Проверить и вернуть поддерживаемое вычислительное устройство."""

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна; проверьте CUDA-сборку PyTorch и драйвер NVIDIA")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Поддерживаются только устройства cuda и cpu")
    return device


def training_config_payload(config: TrainingConfig) -> dict[str, object]:
    """Сериализовать воспроизводимые параметры запуска в checkpoint."""

    return {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "epochs": config.epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "gradient_clip_norm": config.gradient_clip_norm,
        "validation_interval": config.validation_interval,
        "checkpoint_interval": config.checkpoint_interval,
        "validation_batches": config.validation_batches,
        "log_interval": config.log_interval,
        "random_seed": config.random_seed,
        "mixed_precision": config.mixed_precision,
        "model_backend": config.model_backend,
        "pretrained_config_file": (
            None if config.pretrained_config_file is None else str(config.pretrained_config_file)
        ),
    }


def save_named_checkpoint(
    name: str,
    *,
    config: TrainingConfig,
    model: CausalLMBackend,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    state: TrainingState,
) -> Path:
    """Сохранить checkpoint с заданным именем в каталоге текущего запуска."""

    path = config.checkpoint_dir / name
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        training_config=training_config_payload(config),
        state=state,
    )
    return path
