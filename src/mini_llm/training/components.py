"""Фабрика model/tokenizer компонентов для универсального trainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_llm.data.interfaces import DialogueTokenizer
from mini_llm.modeling import CustomCausalLMBackend, DecoderOnlyTransformer, ModelConfig
from mini_llm.modeling.interface import CausalLMBackend
from mini_llm.observability import terminal_log
from mini_llm.pretrained import PretrainedConfig, prepare_pretrained_model
from mini_llm.tokenization import BPETokenizer
from mini_llm.training.config import TrainingConfig
from mini_llm.training.precision import validate_pretrained_training_device


@dataclass(frozen=True, slots=True)
class TrainingComponents:
    """Model backend и tokenizer одного training run."""

    model: CausalLMBackend
    tokenizer: DialogueTokenizer


def create_training_components(
    config: TrainingConfig,
    device: torch.device,
) -> TrainingComponents:
    """Создать custom либо pretrained компоненты по конфигурации запуска."""

    if config.model_backend == "pretrained":
        config_file = config.pretrained_config_file
        if config_file is None:
            raise RuntimeError("Не указан pretrained_config_file")
        pretrained_config = PretrainedConfig.load(config_file)
        if pretrained_config.torch_dtype == "auto":
            # Для обучения auto явно фиксируется в BF16; inference-конфиг не меняется.
            pretrained_config = pretrained_config.model_copy(update={"torch_dtype": "bfloat16"})
        terminal_log(
            "PARAMS",
            f"model_id={pretrained_config.model_id} mode={pretrained_config.adaptation_mode} "
            f"dtype={pretrained_config.torch_dtype} "
            f"seq_length={pretrained_config.max_sequence_length} batch_size={config.batch_size} "
            f"gradient_checkpointing={pretrained_config.gradient_checkpointing} "
            f"optimizer={'AdamW8bit' if pretrained_config.adaptation_mode == 'qlora' else 'AdamW'} "
            f"max_train_samples={config.max_train_samples or 'all'} "
            f"num_workers={config.num_workers}",
        )
        validate_pretrained_training_device(
            pretrained_config, device, mixed_precision=config.mixed_precision
        )
        pretrained = prepare_pretrained_model(pretrained_config)
        pretrained.model.to(device)
        return TrainingComponents(pretrained.model, pretrained.tokenizer)

    tokenizer = BPETokenizer.load(config.tokenizer_file)
    model_config = ModelConfig()
    if tokenizer.vocab_size != model_config.vocab_size:
        raise RuntimeError(
            f"Vocabulary tokenizer ({tokenizer.vocab_size}) не совпадает с ModelConfig "
            f"({model_config.vocab_size})"
        )
    model = DecoderOnlyTransformer(model_config)
    backend = CustomCausalLMBackend(model, model_config)
    backend.to(device)
    return TrainingComponents(backend, tokenizer)
