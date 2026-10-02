"""Расчёт плана обучения и компактное представление его прогресса."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mini_llm.data.dataset import DialogueDataset
from mini_llm.observability import terminal_log


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """Число шагов в выбранном режиме обучения."""

    steps_per_epoch: int
    planned_total_steps: int
    planned_epochs: int


def calculate_training_plan(
    train_samples: int,
    batch_size: int,
    *,
    epochs: int | None,
    max_steps: int | None,
) -> TrainingPlan:
    """Рассчитать план для взаимоисключающих режимов epochs и max_steps."""

    if train_samples <= 0:
        raise ValueError("train_samples должен быть положительным")
    if batch_size <= 0:
        raise ValueError("batch_size должен быть положительным")
    if (epochs is None) == (max_steps is None):
        raise ValueError("Нужно задать ровно один режим: epochs или max_steps")
    steps_per_epoch = math.ceil(train_samples / batch_size)
    if epochs is not None:
        if epochs <= 0:
            raise ValueError("epochs должен быть положительным")
        return TrainingPlan(steps_per_epoch, steps_per_epoch * epochs, epochs)
    if max_steps is None or max_steps <= 0:
        raise ValueError("max_steps должен быть положительным")
    return TrainingPlan(
        steps_per_epoch,
        max_steps,
        math.ceil(max_steps / steps_per_epoch),
    )


def rolling_average(losses: list[float], window_size: int = 100) -> float:
    """Вернуть средний loss по последнему окну шагов."""

    if not losses:
        return math.nan
    window = losses[-window_size:]
    return sum(window) / len(window)


def format_training_progress(
    *,
    epoch: int,
    planned_epochs: int,
    epoch_progress: float,
    global_step: int,
    planned_total_steps: int,
    samples_seen: int,
    tokens_seen: int,
    train_loss: float,
    rolling_loss: float,
    learning_rate: float,
    tokens_per_second: float,
    telemetry: str,
) -> str:
    """Собрать одну стабильную строку прогресса для терминала и тестов."""

    return (
        f"epoch={epoch}/{planned_epochs} epoch_progress={epoch_progress:.2f}% "
        f"step={global_step}/{planned_total_steps} samples_seen={samples_seen} "
        f"tokens_seen={tokens_seen} train_loss={train_loss:.6f} "
        f"rolling_train_loss_100={rolling_loss:.6f} lr={learning_rate:.6g} "
        f"tokens/sec={tokens_per_second:.1f} {telemetry}"
    )


def effective_train_tokens(
    statistics_file: Path,
    dataset: DialogueDataset,
    *,
    max_sequence_length: int,
) -> tuple[int | None, str]:
    """Прочитать совместимую статистику без пересчёта токенов по Dataset."""

    if statistics_file.is_file():
        try:
            payload_object = cast(object, json.loads(statistics_file.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload_object = None
        if isinstance(payload_object, dict):
            payload = cast(dict[str, object], payload_object)
            splits = payload.get("splits")
            train = splits.get("train") if isinstance(splits, dict) else None
            if (
                payload.get("max_sequence_length") == max_sequence_length
                and payload.get("training_objective") == "response_only"
                and isinstance(train, dict)
                and train.get("usable_samples") == len(dataset)
                and isinstance(train.get("training_loss_tokens"), int)
            ):
                token_count = cast(int, train["training_loss_tokens"])
                terminal_log(
                    "TOKENS",
                    f"Effective train tokens прочитаны из statistics JSON: "
                    f"file={statistics_file} training_loss_tokens={token_count}",
                )
                return token_count, str(statistics_file)

    terminal_log(
        "TOKENS",
        "Совместимая response-only статистика отсутствует; effective_train_tokens=unknown. "
        "Предварительный пересчёт отключён, обучение продолжается без сканирования токенов.",
    )
    return None, "unknown"
