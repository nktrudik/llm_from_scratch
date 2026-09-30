"""Сохранение и восстановление полного состояния training run."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.optim import Optimizer

from mini_llm.config import ModelConfig
from mini_llm.model import DecoderOnlyTransformer


@dataclass(slots=True)
class TrainingState:
    """Позиция training run, необходимая для точного resume."""

    epoch: int = 0
    batches_completed_in_epoch: int = 0
    global_step: int = 0
    best_validation_loss: float = math.inf
    last_train_loss: float = math.nan
    last_validation_loss: float | None = None


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"Checkpoint field {name} должен быть object")
    return cast(dict[str, object], value)


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"Checkpoint field {name} должен быть integer")
    return value


def _require_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"Checkpoint field {name} должен быть number")
    return float(value)


def save_checkpoint(
    path: Path,
    *,
    model: DecoderOnlyTransformer,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    model_config: ModelConfig,
    training_config: Mapping[str, object],
    state: TrainingState,
) -> None:
    """Атомарно сохранить полное состояние для resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "format_version": 1,
        "model_config": cast(dict[str, object], asdict(model_config)),
        "training_config": dict(training_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": state.epoch,
        "batches_completed_in_epoch": state.batches_completed_in_epoch,
        "global_step": state.global_step,
        "best_validation_loss": state.best_validation_loss,
        "last_train_loss": state.last_train_loss,
        "last_validation_loss": state.last_validation_loss,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: DecoderOnlyTransformer,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    model_config: ModelConfig,
    learning_rate: float,
) -> TrainingState:
    """Восстановить model/optimizer/scaler/RNG и позицию training run."""

    if not path.is_file():
        raise RuntimeError(f"Checkpoint не найден: {path}")
    try:
        payload_object = cast(
            object,
            torch.load(path, map_location="cpu", weights_only=True),
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"Не удалось загрузить checkpoint {path}: {error}") from error
    payload = _require_mapping(payload_object, "root")
    if _require_int(payload.get("format_version"), "format_version") != 1:
        raise RuntimeError("Неподдерживаемая версия checkpoint")
    saved_model_config = _require_mapping(payload.get("model_config"), "model_config")
    expected_model_config = cast(dict[str, object], asdict(model_config))
    if saved_model_config != expected_model_config:
        raise RuntimeError("ModelConfig checkpoint не совпадает с текущей конфигурацией")

    model_state = cast(Mapping[str, Tensor], payload.get("model_state_dict"))
    optimizer_state = _require_mapping(payload.get("optimizer_state_dict"), "optimizer_state_dict")
    scaler_state = _require_mapping(payload.get("scaler_state_dict"), "scaler_state_dict")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    scaler.load_state_dict(scaler_state)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    rng_state = payload.get("torch_rng_state")
    if not isinstance(rng_state, Tensor):
        raise RuntimeError("Checkpoint field torch_rng_state должен быть Tensor")
    torch.set_rng_state(rng_state)
    cuda_rng_states = payload.get("cuda_rng_state_all")
    if torch.cuda.is_available() and isinstance(cuda_rng_states, list):
        torch.cuda.set_rng_state_all(cast(list[Tensor], cuda_rng_states))

    validation_value = payload.get("last_validation_loss")
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
    )
