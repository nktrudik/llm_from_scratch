"""Явный выбор точности обучения без скрытой замены BF16 на FP16."""

from __future__ import annotations

import torch

from mini_llm.pretrained.config import PretrainedConfig


def validate_pretrained_training_device(
    config: PretrainedConfig, device: torch.device, *, mixed_precision: bool
) -> None:
    """Проверить устройство и поддержку BF16 до загрузки весов."""

    if config.adaptation_mode == "qlora" and device.type != "cuda":
        raise RuntimeError("QLoRA training с AdamW8bit требует CUDA GPU")
    requires_bf16 = config.torch_dtype == "bfloat16" or (
        mixed_precision and config.torch_dtype != "float16"
    )
    if device.type == "cuda" and requires_bf16:
        with torch.cuda.device(device):
            supported = torch.cuda.is_bf16_supported(including_emulation=False)
        if not supported:
            raise RuntimeError(
                "GPU не поддерживает аппаратный BF16. Автоматический fallback отключён; "
                "явно задайте torch_dtype=float16 в pretrained config (--dtype float16 в setup)."
            )


def create_grad_scaler(*, use_amp: bool, amp_dtype: torch.dtype) -> torch.amp.GradScaler:
    """Масштабировать градиенты только для FP16; BF16 не требует GradScaler."""

    return torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
