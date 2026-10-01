"""Последовательный pipeline обучения, validation и сохранения checkpoints."""

from __future__ import annotations

import time

import torch
from torch.optim import AdamW

from mini_llm.data.dataset import IGNORE_INDEX, DataLoaderConfig, DialogueDataset, create_dataloader
from mini_llm.observability import ProgressThrottle, terminal_log, terminal_stage
from mini_llm.training.checkpoints import load_checkpoint
from mini_llm.training.components import create_training_components
from mini_llm.training.config import TrainingConfig
from mini_llm.training.monitoring import evaluate_validation_loss, gpu_telemetry
from mini_llm.training.progress import (
    calculate_training_plan,
    effective_train_tokens,
    format_training_progress,
    rolling_average,
)
from mini_llm.training.runtime import save_named_checkpoint, select_device
from mini_llm.training.schemas import TrainingResult, TrainingState


def train_model(config: TrainingConfig | None = None) -> TrainingResult:
    """Запустить обучение, validation и checkpointing на одном устройстве."""

    training_started_at = time.perf_counter()
    active_config = config or TrainingConfig()
    terminal_log(
        "TRAINING",
        f"Pipeline запущен backend={active_config.model_backend} "
        f"checkpoint_dir={active_config.checkpoint_dir}",
    )
    with terminal_stage("DEVICE", f"выбор устройства requested={active_config.device}"):
        device = select_device(active_config.device)
        use_amp = active_config.mixed_precision and device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    terminal_log("DEVICE", f"Выбрано device={device} name={gpu_name} AMP={use_amp}")
    torch.manual_seed(active_config.random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(active_config.random_seed)
        torch.cuda.reset_peak_memory_stats(device)

    with terminal_stage(
        "MODEL",
        f"загрузка/подготовка backend={active_config.model_backend} и tokenizer",
    ):
        components = create_training_components(active_config, device)
    tokenizer = components.tokenizer
    model = components.model
    total_parameters = sum(parameter.numel() for parameter in model.module.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.module.parameters() if parameter.requires_grad
    )
    terminal_log(
        "MODEL",
        f"Backend готов: backend={active_config.model_backend} "
        f"total_parameters={total_parameters:,} trainable_parameters={trainable_parameters:,}",
    )
    train_file = active_config.splits_dir / "train.jsonl"
    with terminal_stage("DATASET", f"создание train Dataset file={train_file}"):
        train_dataset = DialogueDataset(
            train_file,
            tokenizer,
            max_sequence_length=model.max_sequence_length,
        )
    validation_file = active_config.splits_dir / "validation.jsonl"
    with terminal_stage("DATASET", f"создание validation Dataset file={validation_file}"):
        validation_dataset = DialogueDataset(
            validation_file,
            tokenizer,
            max_sequence_length=model.max_sequence_length,
        )
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise RuntimeError("Train и validation Dataset должны содержать пригодные samples")
    with terminal_stage("PLAN", "расчёт training plan"):
        plan = calculate_training_plan(
            len(train_dataset),
            active_config.batch_size,
            epochs=active_config.epochs,
            max_steps=active_config.max_steps,
        )
    terminal_log(
        "PLAN",
        f"steps_per_epoch={plan.steps_per_epoch} "
        f"planned_total_steps={plan.planned_total_steps} planned_epochs={plan.planned_epochs}",
    )
    with terminal_stage("TOKENS", "расчёт effective train tokens"):
        train_token_count, token_count_source = effective_train_tokens(
            active_config.token_statistics_file,
            train_dataset,
            max_sequence_length=model.max_sequence_length,
        )

    loader_config = DataLoaderConfig(
        batch_size=active_config.batch_size,
        num_workers=active_config.num_workers,
        random_seed=active_config.random_seed,
        pin_memory=device.type == "cuda",
    )
    with terminal_stage("DATALOADER", "создание validation DataLoader"):
        validation_loader = create_dataloader(validation_dataset, loader_config, shuffle=False)
    with terminal_stage("OPTIMIZER", "создание AdamW и AMP GradScaler"):
        optimizer = AdamW(
            model.trainable_parameters(),
            lr=active_config.learning_rate,
            weight_decay=active_config.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    state = TrainingState()
    if active_config.resume_from is not None:
        with terminal_stage(
            "CHECKPOINT", f"загрузка resume checkpoint path={active_config.resume_from}"
        ):
            state = load_checkpoint(
                active_config.resume_from,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                learning_rate=active_config.learning_rate,
                expected_batch_size=active_config.batch_size,
            )
        terminal_log(
            "CHECKPOINT",
            f"Resume восстановлен step={state.global_step} epoch={state.epoch + 1} "
            f"batches_completed={state.batches_completed_in_epoch}",
        )
    if state.global_step > plan.planned_total_steps:
        raise RuntimeError(
            "Checkpoint находится дальше planned_total_steps; увеличьте epochs или max_steps"
        )
    if state.epoch >= plan.planned_epochs and state.global_step < plan.planned_total_steps:
        raise RuntimeError("Позиция эпохи в checkpoint несовместима с текущим планом")

    terminal_log(
        "TRAINING",
        f"Training start: device={device} ({gpu_name}), AMP={use_amp}, "
        f"train_samples={len(train_dataset)}, validation_samples={len(validation_dataset)}, "
        f"batch_size={active_config.batch_size}, max_sequence_length="
        f"{model.max_sequence_length}, backend={active_config.model_backend}, "
        f"start_step={state.global_step}",
    )
    terminal_log(
        "TRAINING",
        f"Training plan: steps_per_epoch={plan.steps_per_epoch}, "
        f"planned_total_steps={plan.planned_total_steps}, "
        f"train_samples={len(train_dataset)}, effective_train_tokens={train_token_count} "
        f"(source={token_count_source})",
    )
    last_validation_step = -1
    interrupted = False
    stop_requested = False
    loss_since_log = 0.0
    steps_since_log = 0
    tokens_since_log = 0
    log_started = time.perf_counter()

    try:
        for epoch in range(state.epoch, plan.planned_epochs):
            epoch_started_at = time.perf_counter()
            skip_batches = state.batches_completed_in_epoch if epoch == state.epoch else 0
            terminal_log(
                "EPOCH",
                f"Начало epoch={epoch + 1}/{plan.planned_epochs} "
                f"steps_per_epoch={plan.steps_per_epoch} resume_skip_batches={skip_batches}",
            )
            epoch_loader_config = DataLoaderConfig(
                batch_size=active_config.batch_size,
                num_workers=active_config.num_workers,
                random_seed=active_config.random_seed + epoch,
                pin_memory=device.type == "cuda",
            )
            with terminal_stage("DATALOADER", f"создание train DataLoader для epoch={epoch + 1}"):
                train_loader = create_dataloader(train_dataset, epoch_loader_config, shuffle=True)
            model.train()
            first_batch_logged = False
            skip_throttle = ProgressThrottle(every_items=500, every_seconds=5.0)
            terminal_log("TRAIN", f"Ожидание первого batch для epoch={epoch + 1}")
            for batch_index, batch in enumerate(train_loader):
                if batch_index < skip_batches:
                    skipped = batch_index + 1
                    if skip_throttle.should_report(skipped):
                        terminal_log(
                            "TRAIN",
                            f"Resume: пропуск уже обработанных batches={skipped}/{skip_batches}",
                            elapsed=time.perf_counter() - epoch_started_at,
                        )
                    continue
                if state.global_step >= plan.planned_total_steps:
                    stop_requested = True
                    break
                if not first_batch_logged:
                    response_tokens = int((batch["labels"] != IGNORE_INDEX).sum().item())
                    terminal_log(
                        "TRAIN",
                        f"Первый batch получен epoch={epoch + 1} batch={batch_index + 1} "
                        f"shape={tuple(batch['input_ids'].shape)} "
                        f"response_tokens={response_tokens}",
                        elapsed=time.perf_counter() - epoch_started_at,
                    )
                    first_batch_logged = True
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=model.autocast_dtype,
                    enabled=use_amp,
                ):
                    output = model.forward_batch(input_ids, attention_mask, labels)
                torch.autograd.backward(scaler.scale(output.loss))
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), active_config.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()

                state.global_step += 1
                state.epoch = epoch
                state.batches_completed_in_epoch = batch_index + 1
                state.last_train_loss = output.loss.item()
                batch_samples = int(input_ids.size(0))
                batch_tokens = int((labels != IGNORE_INDEX).sum().item())
                state.samples_seen += batch_samples
                state.tokens_seen += batch_tokens
                state.recent_train_losses.append(state.last_train_loss)
                del state.recent_train_losses[:-100]
                loss_since_log += state.last_train_loss
                steps_since_log += 1
                tokens_since_log += batch_tokens

                should_log = state.global_step == 1 or (
                    state.global_step % active_config.log_interval == 0
                )
                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = max(time.perf_counter() - log_started, 1e-9)
                    learning_rate = float(optimizer.param_groups[0]["lr"])
                    terminal_log(
                        "TRAIN",
                        format_training_progress(
                            epoch=epoch + 1,
                            planned_epochs=plan.planned_epochs,
                            epoch_progress=(batch_index + 1) / plan.steps_per_epoch * 100.0,
                            global_step=state.global_step,
                            planned_total_steps=plan.planned_total_steps,
                            samples_seen=state.samples_seen,
                            tokens_seen=state.tokens_seen,
                            train_loss=loss_since_log / steps_since_log,
                            rolling_loss=rolling_average(state.recent_train_losses),
                            learning_rate=learning_rate,
                            tokens_per_second=tokens_since_log / elapsed,
                            telemetry=gpu_telemetry(device),
                        ),
                    )
                    loss_since_log = 0.0
                    steps_since_log = 0
                    tokens_since_log = 0
                    log_started = time.perf_counter()

                if state.global_step % active_config.validation_interval == 0:
                    with terminal_stage("VALIDATION", f"validation на step={state.global_step}"):
                        validation_loss = evaluate_validation_loss(
                            model,
                            validation_loader,
                            device=device,
                            use_amp=use_amp,
                            amp_dtype=model.autocast_dtype,
                            max_batches=active_config.validation_batches,
                        )
                    state.last_validation_loss = validation_loss
                    last_validation_step = state.global_step
                    terminal_log(
                        "VALIDATION",
                        f"validation step={state.global_step} "
                        f"validation_loss={validation_loss:.6f} {gpu_telemetry(device)}",
                    )
                    if validation_loss < state.best_validation_loss:
                        state.best_validation_loss = validation_loss
                        best_path = save_named_checkpoint(
                            "best.pt",
                            config=active_config,
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            state=state,
                        )
                        terminal_log("CHECKPOINT", f"Best checkpoint обновлён: {best_path}")

                if state.global_step % active_config.checkpoint_interval == 0:
                    checkpoint_path = save_named_checkpoint(
                        f"step_{state.global_step:08d}.pt",
                        config=active_config,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        state=state,
                    )
                    terminal_log("CHECKPOINT", f"Periodic checkpoint готов: {checkpoint_path}")

                if state.global_step >= plan.planned_total_steps:
                    stop_requested = True
                    break

            if stop_requested:
                break
            state.epoch = epoch + 1
            state.batches_completed_in_epoch = 0
            terminal_log(
                "EPOCH",
                f"Epoch завершена epoch={epoch + 1}/{plan.planned_epochs} "
                f"global_step={state.global_step}",
                elapsed=time.perf_counter() - epoch_started_at,
            )
    except KeyboardInterrupt:
        interrupted = True
        terminal_log("TRAINING", "Получен KeyboardInterrupt; сохраняется last checkpoint")

    if not interrupted and state.global_step > 0 and last_validation_step != state.global_step:
        with terminal_stage("VALIDATION", f"финальная validation на step={state.global_step}"):
            validation_loss = evaluate_validation_loss(
                model,
                validation_loader,
                device=device,
                use_amp=use_amp,
                amp_dtype=model.autocast_dtype,
                max_batches=active_config.validation_batches,
            )
        state.last_validation_loss = validation_loss
        if validation_loss < state.best_validation_loss:
            state.best_validation_loss = validation_loss
            save_named_checkpoint(
                "best.pt",
                config=active_config,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                state=state,
            )
        terminal_log("VALIDATION", f"final validation_loss={validation_loss:.6f}")

    last_checkpoint = save_named_checkpoint(
        "last.pt",
        config=active_config,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        state=state,
    )
    terminal_log(
        "TRAINING",
        f"Training завершён: step={state.global_step}, "
        f"best_validation_loss={state.best_validation_loss:.6f}, "
        f"last_checkpoint={last_checkpoint}, interrupted={interrupted}",
        elapsed=time.perf_counter() - training_started_at,
    )
    return TrainingResult(
        state.global_step,
        state.best_validation_loss,
        last_checkpoint,
        interrupted,
    )
