"""Сохранение и восстановление полного состояния обучения."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.optim import Optimizer

from mini_llm.modeling import CausalLMBackend
from mini_llm.training.schemas import TrainingState


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"Поле checkpoint {name} должно быть объектом")
    return cast(dict[str, object], value)


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"Поле checkpoint {name} должно быть целым числом")
    return value


def _require_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"Поле checkpoint {name} должно быть числом")
    return float(value)


def save_checkpoint(
    path: Path,
    *,
    model: CausalLMBackend,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    training_config: Mapping[str, object],
    state: TrainingState,
) -> None:
    """Атомарно сохранить полное состояние для последующего продолжения."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "format_version": 3,
        "model_metadata": model.checkpoint_metadata,
        "training_config": dict(training_config),
        "model_state_dict": model.checkpoint_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_name": type(optimizer).__name__,
        "scaler_state_dict": scaler.state_dict(),
        "epoch": state.epoch,
        "batches_completed_in_epoch": state.batches_completed_in_epoch,
        "global_step": state.global_step,
        "best_validation_loss": state.best_validation_loss,
        "last_train_loss": state.last_train_loss,
        "last_validation_loss": state.last_validation_loss,
        "samples_seen": state.samples_seen,
        "tokens_seen": state.tokens_seen,
        "recent_train_losses": state.recent_train_losses[-100:],
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    custom_config = model.checkpoint_metadata.get("model_config")
    if isinstance(custom_config, dict):
        # Поле сохранено для совместимости существующего custom inference.
        payload["model_config"] = custom_config
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: CausalLMBackend,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    learning_rate: float,
    expected_batch_size: int,
    expected_max_train_samples: int | None = None,
) -> TrainingState:
    """Восстановить модель, optimizer, scaler, RNG и позицию обучения."""

    if not path.is_file():
        raise RuntimeError(f"Checkpoint не найден: {path}")
    try:
        payload_object = cast(object, torch.load(path, map_location="cpu", weights_only=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"Не удалось загрузить checkpoint {path}: {error}") from error
    payload = _require_mapping(payload_object, "root")
    format_version = _require_int(payload.get("format_version"), "format_version")
    if format_version not in {1, 2, 3}:
        raise RuntimeError("Неподдерживаемая версия checkpoint")
    if format_version < 3:
        saved_metadata: dict[str, object] = {
            "backend": "custom",
            "model_config": _require_mapping(payload.get("model_config"), "model_config"),
        }
    else:
        saved_metadata = _require_mapping(payload.get("model_metadata"), "model_metadata")
    if saved_metadata != model.checkpoint_metadata:
        raise RuntimeError("Конфигурация model backend в checkpoint не совпадает с текущей")

    saved_training_config = _require_mapping(payload.get("training_config"), "training_config")
    saved_batch_size = _require_int(saved_training_config.get("batch_size"), "batch_size")
    if saved_batch_size != expected_batch_size:
        raise RuntimeError(
            "batch_size checkpoint не совпадает с текущим; "
            "для точного продолжения размер batch менять нельзя"
        )
    saved_optimizer_name = payload.get("optimizer_name", "AdamW")
    if saved_optimizer_name != type(optimizer).__name__:
        raise RuntimeError(
            f"Optimizer checkpoint ({saved_optimizer_name}) не совпадает с текущим "
            f"({type(optimizer).__name__}); состояния AdamW и AdamW8bit не взаимозаменяемы"
        )
    if saved_training_config.get("max_train_samples") != expected_max_train_samples:
        raise RuntimeError(
            "max_train_samples checkpoint не совпадает с текущим; "
            "для точного resume нельзя менять состав train Dataset"
        )

    model_state = cast(Mapping[str, Tensor], payload.get("model_state_dict"))
    optimizer_state = _require_mapping(payload.get("optimizer_state_dict"), "optimizer_state_dict")
    scaler_state = _require_mapping(payload.get("scaler_state_dict"), "scaler_state_dict")
    model.load_checkpoint_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    scaler.load_state_dict(scaler_state)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    rng_state = payload.get("torch_rng_state")
    if not isinstance(rng_state, Tensor):
        raise RuntimeError("Поле checkpoint torch_rng_state должно быть Tensor")
    torch.set_rng_state(rng_state)
    cuda_rng_states = payload.get("cuda_rng_state_all")
    if torch.cuda.is_available() and isinstance(cuda_rng_states, list):
        torch.cuda.set_rng_state_all(cast(list[Tensor], cuda_rng_states))

    validation_value = payload.get("last_validation_loss")
    recent_losses_value = payload.get("recent_train_losses", [])
    if not isinstance(recent_losses_value, list):
        raise RuntimeError("Поле checkpoint recent_train_losses должно быть списком")
    recent_losses = [
        _require_float(value, "recent_train_losses item") for value in recent_losses_value
    ][-100:]
    return TrainingState(
        epoch=_require_int(payload.get("epoch"), "epoch"),
        batches_completed_in_epoch=_require_int(
            payload.get("batches_completed_in_epoch"), "batches_completed_in_epoch"
        ),
        global_step=_require_int(payload.get("global_step"), "global_step"),
        best_validation_loss=_require_float(
            payload.get("best_validation_loss"), "best_validation_loss"
        ),
        last_train_loss=_require_float(payload.get("last_train_loss"), "last_train_loss"),
        last_validation_loss=(
            None
            if validation_value is None
            else _require_float(validation_value, "last_validation_loss")
        ),
        samples_seen=_require_int(payload.get("samples_seen", 0), "samples_seen"),
        tokens_seen=_require_int(payload.get("tokens_seen", 0), "tokens_seen"),
        recent_train_losses=recent_losses,
    )
