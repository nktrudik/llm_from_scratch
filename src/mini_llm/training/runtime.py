"""Вспомогательные операции окружения и checkpoint-файлов."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.optim import Optimizer

from mini_llm.modeling import CausalLMBackend
from mini_llm.observability import terminal_log
from mini_llm.observability.training import report_stage
from mini_llm.training.checkpoints import save_checkpoint
from mini_llm.training.config import TrainingConfig
from mini_llm.training.schemas import TrainingState
from mini_llm.workspace.catalog import ModelRun, update_progress


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

    payload: dict[str, object] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "max_train_samples": config.max_train_samples,
        "max_validation_samples": config.max_validation_samples,
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
        "pretrained_engine": config.pretrained_engine,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_steps": config.warmup_steps,
        "packing": config.packing,
        "splits_dir": config.splits_dir.as_posix(),
        "pretrained_config_file": (
            None if config.pretrained_config_file is None else str(config.pretrained_config_file)
        ),
    }
    manifest = config.checkpoint_dir / "run.json"
    if manifest.is_file():
        run = ModelRun.model_validate_json(manifest.read_text(encoding="utf-8"))
        payload.update(
            run_id=run.run_id,
            dataset_id=run.dataset_id,
            dataset_version=run.dataset_version,
            splits_dir=config.splits_dir.as_posix(),
        )
    return payload


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
    started_at = time.perf_counter()
    terminal_log("CHECKPOINT", f"Начало сохранения name={name} path={path}")
    try:
        with report_stage(f"Сохранение checkpoint {name}"):
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                training_config=training_config_payload(config),
                state=state,
            )
            update_progress(
                config.checkpoint_dir,
                state.global_step,
                checkpoint_name=name,
                validation_loss=state.last_validation_loss,
            )
    except Exception:
        terminal_log(
            "CHECKPOINT",
            f"Сохранение завершилось ошибкой name={name} path={path}",
            elapsed=time.perf_counter() - started_at,
        )
        raise
    terminal_log(
        "CHECKPOINT",
        f"Сохранение завершено name={name} path={path}",
        elapsed=time.perf_counter() - started_at,
    )
    return path
