"""Схемы результата и текущего состояния обучения."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TrainingState:
    """Позиция запуска, необходимая для точного продолжения обучения."""

    epoch: int = 0
    batches_completed_in_epoch: int = 0
    global_step: int = 0
    best_validation_loss: float = math.inf
    last_train_loss: float = math.nan
    last_validation_loss: float | None = None
    samples_seen: int = 0
    tokens_seen: int = 0
    recent_train_losses: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Краткий результат завершившегося или прерванного запуска."""

    global_step: int
    best_validation_loss: float
    last_checkpoint: Path
    interrupted: bool
