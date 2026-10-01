"""Расчёт плана обучения и компактное представление его прогресса."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mini_llm.data.dataset import IGNORE_INDEX, DialogueDataset
from mini_llm.observability import ProgressThrottle, terminal_log


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
) -> tuple[int, str]:
    """Прочитать актуальную статистику или посчитать effective tokens по Dataset."""

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

    total_samples = len(dataset)
    terminal_log(
        "TOKENS",
        "Готовая response-only статистика не найдена; начинается fallback full Dataset scan "
        f"samples={total_samples}",
    )
    started_at = time.perf_counter()
    throttle = ProgressThrottle(every_items=5_000, every_seconds=5.0)
    token_count = 0
    for index in range(total_samples):
        token_count += int((dataset[index]["labels"] != IGNORE_INDEX).sum().item())
        processed = index + 1
        if throttle.should_report(processed):
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            percentage = 100.0 * processed / total_samples if total_samples else 100.0
            terminal_log(
                "TOKENS",
                f"Fallback scan processed={processed}/{total_samples} "
                f"progress={percentage:.2f}% "
                f"speed={processed / elapsed:.1f} samples/sec "
                f"training_loss_tokens={token_count}",
                elapsed=elapsed,
            )
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    terminal_log(
        "TOKENS",
        f"Fallback scan завершён processed={total_samples}/{total_samples} "
        f"progress=100.00% speed={total_samples / elapsed:.1f} samples/sec "
        f"training_loss_tokens={token_count}",
        elapsed=elapsed,
    )
    return token_count, "расчёт по train Dataset"
