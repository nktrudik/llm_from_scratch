"""Последовательный pipeline обучения, validation и сохранения checkpoints."""

from __future__ import annotations

import time

import torch
from torch.optim import AdamW

from mini_llm.data.dataset import DataLoaderConfig, DialogueDataset, create_dataloader
from mini_llm.modeling import DecoderOnlyTransformer, ModelConfig
from mini_llm.tokenization import BPETokenizer
from mini_llm.training.checkpoints import load_checkpoint
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

    active_config = config or TrainingConfig()
    device = select_device(active_config.device)
    use_amp = active_config.mixed_precision and device.type == "cuda"
    torch.manual_seed(active_config.random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(active_config.random_seed)
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer = BPETokenizer.load(active_config.tokenizer_file)
    model_config = ModelConfig()
    if tokenizer.vocab_size != model_config.vocab_size:
        raise RuntimeError(
            f"Vocabulary tokenizer ({tokenizer.vocab_size}) не совпадает с ModelConfig "
            f"({model_config.vocab_size})"
        )
    train_dataset = DialogueDataset(
        active_config.splits_dir / "train.jsonl",
        tokenizer,
        max_sequence_length=model_config.max_sequence_length,
    )
    validation_dataset = DialogueDataset(
        active_config.splits_dir / "validation.jsonl",
        tokenizer,
        max_sequence_length=model_config.max_sequence_length,
    )
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise RuntimeError("Train и validation Dataset должны содержать пригодные samples")
    plan = calculate_training_plan(
        len(train_dataset),
        active_config.batch_size,
        epochs=active_config.epochs,
        max_steps=active_config.max_steps,
    )
    train_token_count, token_count_source = effective_train_tokens(
        active_config.token_statistics_file,
        train_dataset,
        max_sequence_length=model_config.max_sequence_length,
    )

    loader_config = DataLoaderConfig(
        batch_size=active_config.batch_size,
        num_workers=active_config.num_workers,
        random_seed=active_config.random_seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_dataloader(validation_dataset, loader_config, shuffle=False)
    model = DecoderOnlyTransformer(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=active_config.learning_rate,
        weight_decay=active_config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    state = TrainingState()
    if active_config.resume_from is not None:
        state = load_checkpoint(
            active_config.resume_from,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            model_config=model_config,
            learning_rate=active_config.learning_rate,
            expected_batch_size=active_config.batch_size,
        )
    if state.global_step > plan.planned_total_steps:
        raise RuntimeError(
            "Checkpoint находится дальше planned_total_steps; увеличьте epochs или max_steps"
        )
    if state.epoch >= plan.planned_epochs and state.global_step < plan.planned_total_steps:
        raise RuntimeError("Позиция эпохи в checkpoint несовместима с текущим планом")

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(
        f"Training start: device={device} ({gpu_name}), AMP={use_amp}, "
        f"train_samples={len(train_dataset)}, validation_samples={len(validation_dataset)}, "
        f"batch_size={active_config.batch_size}, max_sequence_length="
        f"{model_config.max_sequence_length}, start_step={state.global_step}"
    )
    print(
        f"Training plan: steps_per_epoch={plan.steps_per_epoch}, "
        f"planned_total_steps={plan.planned_total_steps}, "
        f"train_samples={len(train_dataset)}, effective_train_tokens={train_token_count} "
        f"(source={token_count_source})"
    )
    pad_token_id = model_config.pad_token_id
    last_validation_step = -1
    interrupted = False
    stop_requested = False
    loss_since_log = 0.0
    steps_since_log = 0
    tokens_since_log = 0
    log_started = time.perf_counter()

    try:
        for epoch in range(state.epoch, plan.planned_epochs):
            epoch_loader_config = DataLoaderConfig(
                batch_size=active_config.batch_size,
                num_workers=active_config.num_workers,
                random_seed=active_config.random_seed + epoch,
                pin_memory=device.type == "cuda",
            )
            train_loader = create_dataloader(train_dataset, epoch_loader_config, shuffle=True)
            skip_batches = state.batches_completed_in_epoch if epoch == state.epoch else 0
            model.train()
            for batch_index, batch in enumerate(train_loader):
                if batch_index < skip_batches:
                    continue
                if state.global_step >= plan.planned_total_steps:
                    stop_requested = True
                    break
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_amp,
                ):
                    _, loss = model(input_ids, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), active_config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                state.global_step += 1
                state.epoch = epoch
                state.batches_completed_in_epoch = batch_index + 1
                state.last_train_loss = loss.item()
                batch_samples = int(input_ids.size(0))
                batch_tokens = int((targets != pad_token_id).sum().item())
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
                    print(
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
                        )
                    )
                    loss_since_log = 0.0
                    steps_since_log = 0
                    tokens_since_log = 0
                    log_started = time.perf_counter()

                if state.global_step % active_config.validation_interval == 0:
                    validation_loss = evaluate_validation_loss(
                        model,
                        validation_loader,
                        device=device,
                        use_amp=use_amp,
                        pad_token_id=pad_token_id,
                        max_batches=active_config.validation_batches,
                    )
                    state.last_validation_loss = validation_loss
                    last_validation_step = state.global_step
                    print(
                        f"validation step={state.global_step} "
                        f"validation_loss={validation_loss:.6f} {gpu_telemetry(device)}"
                    )
                    if validation_loss < state.best_validation_loss:
                        state.best_validation_loss = validation_loss
                        best_path = save_named_checkpoint(
                            "best.pt",
                            config=active_config,
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            model_config=model_config,
                            state=state,
                        )
                        print(f"Best checkpoint сохранён: {best_path}")

                if state.global_step % active_config.checkpoint_interval == 0:
                    checkpoint_path = save_named_checkpoint(
                        f"step_{state.global_step:08d}.pt",
                        config=active_config,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        model_config=model_config,
                        state=state,
                    )
                    print(f"Checkpoint сохранён: {checkpoint_path}")

                if state.global_step >= plan.planned_total_steps:
                    stop_requested = True
                    break

            if stop_requested:
                break
            state.epoch = epoch + 1
            state.batches_completed_in_epoch = 0
    except KeyboardInterrupt:
        interrupted = True
        print("Получен KeyboardInterrupt; сохраняется last checkpoint...")

    if not interrupted and state.global_step > 0 and last_validation_step != state.global_step:
        validation_loss = evaluate_validation_loss(
            model,
            validation_loader,
            device=device,
            use_amp=use_amp,
            pad_token_id=pad_token_id,
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
                model_config=model_config,
                state=state,
            )
        print(f"final validation_loss={validation_loss:.6f}")

    last_checkpoint = save_named_checkpoint(
        "last.pt",
        config=active_config,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        model_config=model_config,
        state=state,
    )
    print(
        f"Training завершён: step={state.global_step}, "
        f"best_validation_loss={state.best_validation_loss:.6f}, "
        f"last_checkpoint={last_checkpoint}, interrupted={interrupted}"
    )
    return TrainingResult(
        state.global_step,
        state.best_validation_loss,
        last_checkpoint,
        interrupted,
    )
