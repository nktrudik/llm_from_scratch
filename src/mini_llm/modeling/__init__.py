"""Архитектура и конфигурация decoder-only Transformer."""

from mini_llm.modeling.attention import CausalSelfAttention
from mini_llm.modeling.blocks import MLP, TransformerBlock
from mini_llm.modeling.config import ModelConfig
from mini_llm.modeling.interface import CausalLMBackend, CausalLMOutput, CustomCausalLMBackend
from mini_llm.modeling.transformer import DecoderOnlyTransformer

__all__ = [
    "CausalSelfAttention",
    "CausalLMBackend",
    "CausalLMOutput",
    "CustomCausalLMBackend",
    "DecoderOnlyTransformer",
    "MLP",
    "ModelConfig",
    "TransformerBlock",
]
