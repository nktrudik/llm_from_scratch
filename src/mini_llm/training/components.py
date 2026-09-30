"""Фабрика model/tokenizer компонентов для универсального trainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_llm.data.interfaces import DialogueTokenizer
from mini_llm.modeling import CustomCausalLMBackend, DecoderOnlyTransformer, ModelConfig
from mini_llm.modeling.interface import CausalLMBackend
from mini_llm.pretrained import PretrainedConfig, prepare_pretrained_model
from mini_llm.tokenization import BPETokenizer
from mini_llm.training.config import TrainingConfig


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
        pretrained = prepare_pretrained_model(PretrainedConfig.load(config_file))
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
