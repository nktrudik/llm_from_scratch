"""Явное преобразование профиля проекта в настройки TRL, без скрытых оптимизаций."""

from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.training.config import TrainingConfig


def sft_arguments(
    config: TrainingConfig, model: PretrainedConfig, *, packing: bool | None = None
) -> dict[str, object]:
    """Передать профиль SFT, AMP и настройки точного продолжения DataLoader."""

    cuda_amp = config.device.startswith("cuda") and config.mixed_precision
    return {
        "output_dir": str(config.checkpoint_dir),
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.epochs or 1,
        "max_steps": config.max_steps if config.max_steps is not None else -1,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_steps": config.warmup_steps,
        "optim": "adamw_8bit" if model.adaptation_mode == "qlora" else "adamw_torch",
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.gradient_clip_norm,
        "bf16": cuda_amp and model.torch_dtype != "float16",
        "fp16": cuda_amp and model.torch_dtype == "float16",
        "use_cpu": not config.device.startswith("cuda"),
        "gradient_checkpointing": model.gradient_checkpointing,
        "eval_strategy": "steps",
        "eval_steps": config.validation_interval,
        "save_strategy": "steps",
        "save_steps": config.checkpoint_interval,
        "save_only_model": False,
        "save_total_limit": None,
        "load_best_model_at_end": False,
        "logging_steps": config.log_interval,
        "logging_first_step": True,
        "report_to": "none",
        "dataloader_num_workers": config.num_workers,
        "dataloader_pin_memory": config.device.startswith("cuda"),
        "seed": config.random_seed,
        "data_seed": config.random_seed,
        "ignore_data_skip": False,
        "remove_unused_columns": False,
        "prediction_loss_only": True,
        "max_length": model.max_sequence_length,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "packing": config.packing if packing is None else packing,
        "packing_strategy": "bfd",
        "eval_packing": False,
        "padding_free": False,
        # Маска уже построена единым non-thinking tokenizer, в том числе для Qwen2.5.
        "assistant_only_loss": False,
        "completion_only_loss": False,
    }
