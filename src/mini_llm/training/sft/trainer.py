"""Наблюдаемость и безопасная остановка стандартного SFTTrainer."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from typing import cast

from torch import Tensor, nn
from transformers import TrainerCallback, TrainerControl, TrainerState
from trl.trainer.sft_trainer import SFTTrainer

from mini_llm.observability import terminal_log, terminal_stage
from mini_llm.observability.training import current_training_progress
from mini_llm.pretrained.model import PreparedPretrained
from mini_llm.training.config import TrainingConfig
from mini_llm.training.control import training_stop_requested
from mini_llm.training.monitoring import gpu_telemetry
from mini_llm.training.sft.checkpoints import export_checkpoint
from mini_llm.workspace.catalog import update_progress


class ObservedSFTTrainer(SFTTrainer):
    """Наблюдать loss и реальные токены без изменения его расчёта в TRL."""

    observer: SFTObserver

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, object]:
        loss_tokens = int((inputs["labels"][:, 1:] != -100).sum().item())
        samples = int(inputs["input_ids"].size(0))
        if model.training and not self.observer.first_batch_logged:
            terminal_log(
                "SFT TRAIN",
                f"Первый batch получен: shape={tuple(inputs['input_ids'].shape)} "
                f"response_tokens={loss_tokens}",
            )
            self.observer.first_batch_logged = True
        compute = cast(Callable[..., object], super().compute_loss)
        result = compute(model, inputs, return_outputs, num_items_in_batch)
        loss = result[0] if isinstance(result, tuple) else result
        if model.training and isinstance(loss, Tensor):
            self.observer.losses.append(float(loss.detach().item()))
            self.observer.tokens_seen += loss_tokens
            self.observer.samples_seen += samples
        if isinstance(result, Tensor):
            return result
        if isinstance(result, tuple) and isinstance(result[0], Tensor):
            return cast(tuple[Tensor, object], result)
        raise RuntimeError("SFTTrainer не вернул Tensor loss")

    def evaluate(self, *args: object, **kwargs: object) -> dict[str, float]:
        """Показать отдельный этап validation в терминале и UI."""

        with terminal_stage("SFT VALIDATION", "оценка validation loss"):
            evaluate = cast(Callable[..., dict[str, float]], super().evaluate)
            return evaluate(*args, **kwargs)


class SFTObserver(TrainerCallback):
    """Публиковать progress, экспортировать .pt и передавать запрос остановки trainer-у."""

    def __init__(self, config: TrainingConfig, prepared: PreparedPretrained) -> None:
        self.config = config
        self.prepared = prepared
        self.losses: deque[float] = deque(maxlen=100)
        self.best_loss = math.inf
        self.validation_loss: float | None = None
        self.new_best = False
        self.interrupted = False
        self.samples_seen = self.tokens_seen = 0
        self.started = time.perf_counter()
        self.start_tokens = 0
        self.first_batch_logged = False
        self.checkpoint_started: float | None = None

    def _checkpoint_begin(self, step: int) -> None:
        self.checkpoint_started = time.perf_counter()
        terminal_log(
            "SFT CHECKPOINT", f"Начало сохранения checkpoint-{step} в {self.config.checkpoint_dir}"
        )
        progress = current_training_progress()
        if progress:
            progress.snapshot.stage = f"Сохранение checkpoint-{step}"
            progress.publish(force=True)

    def on_train_begin(
        self, args: object, state: TrainerState, control: TrainerControl, **kwargs: object
    ) -> None:
        """Настроить общий план после восстановления native TrainerState."""

        step = int(state.global_step)
        total = int(state.max_steps)
        progress = current_training_progress()
        if progress:
            progress.snapshot.total_steps = total
            progress.start(step=step, tokens=self.tokens_seen)
        update_progress(self.config.checkpoint_dir, step, total)
        self.started = time.perf_counter()
        self.start_tokens = self.tokens_seen

    def on_step_end(
        self, args: object, state: TrainerState, control: TrainerControl, **kwargs: object
    ) -> None:
        """Останавливать только после законченного optimizer step и обязательно сохранять его."""

        step = int(state.global_step)
        total = int(state.max_steps)
        epoch = float(state.epoch or 0.0)
        self.interrupted = training_stop_requested()
        if self.interrupted:
            control.should_training_stop = True
        if self.interrupted or step >= total:
            control.should_save = True
            control.should_evaluate = not self.interrupted
        progress = current_training_progress()
        if progress:
            snapshot = progress.snapshot
            snapshot.global_step = step
            snapshot.epoch = max(1, math.ceil(epoch))
            snapshot.epoch_progress = ((epoch % 1) or 1.0) * 100
            snapshot.samples_seen = self.samples_seen
            snapshot.tokens_seen = self.tokens_seen
            snapshot.recent_losses = list(self.losses)
            snapshot.train_loss = self.losses[-1] if self.losses else None
            snapshot.rolling_loss = sum(self.losses) / len(self.losses) if self.losses else None
            snapshot.stage = "Остановка и сохранение" if self.interrupted else "Обучение"
            progress.publish()
        if control.should_save and not control.should_evaluate:
            self._checkpoint_begin(step)

    def on_log(
        self,
        args: object,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Сохранить терминальные метрики без внешнего monitoring-сервиса."""

        if not logs:
            return
        progress = current_training_progress()
        lr = logs.get("learning_rate")
        if progress and isinstance(lr, (int, float)):
            progress.snapshot.learning_rate = float(lr)
        if "loss" in logs:
            elapsed = max(time.perf_counter() - self.started, 1e-9)
            terminal_log(
                "SFT TRAIN",
                f"step={state.global_step}/{state.max_steps} "
                f"epoch={state.epoch} train_loss={logs['loss']} "
                f"rolling_loss={sum(self.losses) / len(self.losses) if self.losses else 0:.5f} "
                f"lr={lr} samples_seen={self.samples_seen} tokens_seen={self.tokens_seen} "
                f"tokens/sec={(self.tokens_seen - self.start_tokens) / elapsed:.1f} "
                f"{gpu_telemetry(next(self.prepared.model.module.parameters()).device)}",
            )

    def on_evaluate(
        self,
        args: object,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Сохранить лучший результат даже вне periodic checkpoint interval."""

        value = (metrics or {}).get("eval_loss")
        if isinstance(value, (int, float)) and math.isfinite(value):
            self.validation_loss = float(value)
            self.new_best = value < self.best_loss
            if self.new_best:
                self.best_loss = float(value)
                control.should_save = True
            progress = current_training_progress()
            if progress:
                progress.snapshot.validation_loss = float(value)
                progress.publish(force=True)
            terminal_log("SFT VALIDATION", f"step={state.global_step} loss={value:.6f}")
        if control.should_save:
            self._checkpoint_begin(state.global_step)

    def on_save(
        self, args: object, state: TrainerState, control: TrainerControl, **kwargs: object
    ) -> None:
        """После native save экспортировать inference-совместимые веса и manifest."""

        step = int(state.global_step)
        with terminal_stage("SFT CHECKPOINT", f"экспорт step={step}"):
            export_checkpoint(
                self.config,
                self.prepared,
                step=step,
                loss=self.validation_loss,
                best_loss=self.best_loss,
                best=self.new_best,
                samples_seen=self.samples_seen,
                tokens_seen=self.tokens_seen,
            )
        self.new_best = False
        terminal_log(
            "SFT CHECKPOINT",
            f"Сохранён checkpoint-{step}, last.pt и inference export",
            elapsed=time.perf_counter() - (self.checkpoint_started or time.perf_counter()),
        )
        progress = current_training_progress()
        if progress:
            progress.snapshot.stage = "Обучение"
            progress.publish(force=True)
        if training_stop_requested():
            self.interrupted = True
            control.should_training_stop = True
