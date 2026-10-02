"""Выбор optimizer без зависимости trainer от реализации модели."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from torch.optim import AdamW, Optimizer

from mini_llm.modeling import CausalLMBackend
from mini_llm.pretrained.dependencies import require_module
from mini_llm.training.config import TrainingConfig


def optimizer_name(model: CausalLMBackend) -> str:
    """Определить optimizer по сохранённому описанию backend и режима адаптации."""

    metadata = model.checkpoint_metadata
    pretrained = metadata.get("pretrained_config")
    if (
        metadata.get("backend") == "pretrained"
        and isinstance(pretrained, dict)
        and pretrained.get("adaptation_mode") == "qlora"
    ):
        return "AdamW8bit"
    return "AdamW"


def create_optimizer(model: CausalLMBackend, config: TrainingConfig) -> Optimizer:
    """Использовать bitsandbytes только для QLoRA, сохранив обычный AdamW остальных режимов."""

    if optimizer_name(model) == "AdamW8bit":
        module = require_module("bitsandbytes.optim")
        factory = cast(Callable[..., object], module.AdamW8bit)
        optimizer = factory(
            model.trainable_parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        if not isinstance(optimizer, Optimizer):
            raise RuntimeError("bitsandbytes.AdamW8bit не вернул torch.optim.Optimizer")
        return optimizer
    return AdamW(
        model.trainable_parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
