"""Опциональная подготовка QLoRA через Unsloth без неявного fallback."""

import sys
from collections.abc import Callable
from typing import cast

import torch
from torch import nn

from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.dependencies import require_module
from mini_llm.pretrained.model import HuggingFaceCausalLMBackend, PreparedPretrained
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer


def prepare_unsloth(config: PretrainedConfig, *, random_seed: int) -> PreparedPretrained:
    """Загрузить Unsloth до transformers; при несовместимой среде завершиться явно."""

    if config.adaptation_mode != "qlora":
        raise ValueError("Unsloth-профиль проекта поддерживает QLoRA; для full/LoRA выберите HF")
    if config.qlora_quant_type != "nf4" or not config.qlora_double_quant:
        raise ValueError("Unsloth-профиль требует NF4 + double quant")
    if "transformers" in sys.modules and "unsloth" not in sys.modules:
        raise RuntimeError(
            "Unsloth нужно загрузить до transformers. Перезапустите API и начните обучение "
            "до генерации/загрузки модели либо запустите workspace train в отдельном терминале."
        )
    unsloth = require_module("unsloth")
    factory = unsloth.FastLanguageModel
    load = cast(Callable[..., tuple[object, object]], factory.from_pretrained)
    model, tokenizer = load(
        model_name=config.model_id,
        revision=config.revision,
        cache_dir=str(config.cache_dir),
        local_files_only=config.local_files_only,
        max_seq_length=config.max_sequence_length,
        dtype=torch.float16 if config.torch_dtype == "float16" else torch.bfloat16,
        load_in_4bit=True,
        trust_remote_code=False,
        use_exact_model_name=True,
        fix_tokenizer=False,
        use_gradient_checkpointing=config.gradient_checkpointing,
        random_state=random_seed,
    )
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Unsloth не загрузил 4-bit модель; автоматический fallback запрещён")
    adapt = cast(Callable[..., object], factory.get_peft_model)
    model = adapt(
        model,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=config.gradient_checkpointing,
        random_state=random_seed,
    )
    if not isinstance(model, nn.Module):
        raise RuntimeError("Unsloth не вернул torch.nn.Module")
    return PreparedPretrained(
        HuggingFaceCausalLMBackend(model, config), HuggingFaceDialogueTokenizer(tokenizer)
    )
