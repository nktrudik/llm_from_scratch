"""Публичный API загрузки и адаптации open-source language models."""

from mini_llm.pretrained.config import AdaptationMode, PretrainedConfig
from mini_llm.pretrained.dependencies import PretrainedDependencyError
from mini_llm.pretrained.inference import PretrainedGenerationResult, generate_pretrained
from mini_llm.pretrained.model import (
    HuggingFaceCausalLMBackend,
    PreparedPretrained,
    prepare_pretrained_model,
    save_pretrained_parameters,
)
from mini_llm.pretrained.setup import setup_pretrained_model
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer

__all__ = [
    "AdaptationMode",
    "HuggingFaceCausalLMBackend",
    "HuggingFaceDialogueTokenizer",
    "PreparedPretrained",
    "PretrainedConfig",
    "PretrainedDependencyError",
    "PretrainedGenerationResult",
    "generate_pretrained",
    "prepare_pretrained_model",
    "save_pretrained_parameters",
    "setup_pretrained_model",
]
