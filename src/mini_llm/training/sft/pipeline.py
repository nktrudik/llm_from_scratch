"""Единственный pretrained training pipeline: TRL SFTTrainer и native HF resume."""

import math
from collections.abc import Callable
from importlib.metadata import version
from typing import cast

import torch

from mini_llm.observability import terminal_log, terminal_stage
from mini_llm.observability.training import current_training_progress
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.dependencies import require_module
from mini_llm.pretrained.model import prepare_pretrained_model
from mini_llm.training.config import TrainingConfig
from mini_llm.training.control import TrainingCancelled, training_stop_requested
from mini_llm.training.precision import validate_pretrained_training_device
from mini_llm.training.progress import calculate_training_plan
from mini_llm.training.runtime import select_device
from mini_llm.training.schemas import TrainingResult
from mini_llm.training.sft.checkpoints import resolve_sft_resume, resume_metadata
from mini_llm.training.sft.config import sft_arguments
from mini_llm.training.sft.data import ResponseOnlyCollator, prepare_dataset


def train_pretrained(config: TrainingConfig) -> TrainingResult:
    """Обучать полные non-thinking разговоры; loss только на последнем assistant-ответе."""

    if config.pretrained_config_file is None:
        raise ValueError("Не задан pretrained config")
    if training_stop_requested():
        raise TrainingCancelled
    base = PretrainedConfig.load(config.pretrained_config_file)
    with terminal_stage("SFT DEVICE", f"device={config.device}"):
        device = select_device(config.device)
        validate_pretrained_training_device(base, device, mixed_precision=config.mixed_precision)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            terminal_log("SFT DEVICE", f"GPU: {torch.cuda.get_device_name(device)}")
    resume = resolve_sft_resume(config.resume_from) if config.resume_from else None
    saved = resume_metadata(config, base.model_dump(mode="json"))
    if training_stop_requested():
        raise TrainingCancelled
    torch.manual_seed(config.random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.random_seed)
    with terminal_stage("SFT MODEL", f"model={base.model_id} engine={config.pretrained_engine}"):
        if config.pretrained_engine == "unsloth":
            from mini_llm.training.sft.unsloth import prepare_unsloth

            prepared = prepare_unsloth(base, random_seed=config.random_seed)
        else:
            prepared = prepare_pretrained_model(base)
        prepared.model.to(device)
        total = prepared.total_parameters
        trainable = prepared.trainable_parameters
        terminal_log(
            "SFT MODEL",
            f"Готово: adaptation={base.adaptation_mode} "
            f"total_parameters={total:,} trainable_parameters={trainable:,} "
            f"trainable_percent={100 * trainable / total if total else 0:.3f}%",
        )
    if not prepared.tokenizer.has_chat_template:
        raise ValueError("Для SFT требуется native chat_template instruct-модели")
    train = prepare_dataset(
        config.splits_dir / "train.jsonl",
        prepared.tokenizer,
        max_length=base.max_sequence_length,
        limit=config.max_train_samples,
    )
    validation_limit = config.max_validation_samples
    if config.validation_batches:
        batch_limit = config.validation_batches * config.batch_size
        validation_limit = min(validation_limit, batch_limit) if validation_limit else batch_limit
    validation = prepare_dataset(
        config.splits_dir / "validation.jsonl",
        prepared.tokenizer,
        max_length=base.max_sequence_length,
        limit=validation_limit,
    )
    if training_stop_requested():
        raise TrainingCancelled
    # Импорт после Unsloth: он должен установить свои патчи до transformers/TRL.
    from mini_llm.training.sft.trainer import ObservedSFTTrainer, SFTObserver

    plan = calculate_training_plan(
        len(train), config.batch_size, epochs=config.epochs, max_steps=config.max_steps
    )
    if saved and int(str(saved.get("global_step", 0))) >= plan.planned_total_steps:
        raise ValueError("Checkpoint уже достиг общего плана; увеличьте epochs или max_steps")
    terminal_log(
        "SFT PLAN",
        f"train={config.splits_dir / 'train.jsonl'} samples={len(train)} "
        f"steps_per_epoch={plan.steps_per_epoch} planned_total_steps={plan.planned_total_steps}",
    )
    progress = current_training_progress()
    if progress:
        progress.snapshot.steps_per_epoch = plan.steps_per_epoch
        progress.snapshot.planned_epochs = plan.planned_epochs
    versions = {
        name: version(name) for name in ("torch", "transformers", "trl", "datasets", "peft")
    }
    terminal_log(
        "SFT PARAMS",
        f"engine={config.pretrained_engine} dtype={base.torch_dtype} "
        f"seq_length={base.max_sequence_length} batch_size={config.batch_size} "
        f"gradient_checkpointing={base.gradient_checkpointing} r={base.lora_rank} "
        f"alpha={base.lora_alpha} workers={config.num_workers} thinking=False "
        f"optimizer={'AdamW8bit' if base.adaptation_mode == 'qlora' else 'AdamW'} "
        f"max_train_samples={config.max_train_samples or 'all'} "
        f"versions={versions}",
    )
    factory = cast(Callable[..., object], require_module("trl").SFTConfig)
    arguments = factory(**sft_arguments(config, base))
    observer = SFTObserver(config, prepared)
    with terminal_stage("SFT TRAINER", "создание SFTTrainer"):
        trainer_factory = cast(Callable[..., ObservedSFTTrainer], ObservedSFTTrainer)
        trainer = trainer_factory(
            model=prepared.model.module,
            args=arguments,
            train_dataset=train,
            eval_dataset=validation,
            processing_class=prepared.tokenizer.backend,
            data_collator=ResponseOnlyCollator(prepared.tokenizer.pad_token_id),
            callbacks=[observer],
        )
        trainer.observer = observer
    if saved:
        observer.best_loss = float(str(saved.get("best_validation_loss", math.inf)))
        observer.samples_seen = int(str(saved.get("samples_seen", 0)))
        observer.tokens_seen = int(str(saved.get("tokens_seen", 0)))
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if progress:
        progress.finish(interrupted=observer.interrupted)
    return TrainingResult(
        int(trainer.state.global_step),
        observer.best_loss,
        config.checkpoint_dir / "last.pt",
        observer.interrupted,
    )
