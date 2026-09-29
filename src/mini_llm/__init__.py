"""Public package interface for the compact language model."""

from mini_llm.config import ModelConfig
from mini_llm.model import DecoderOnlyTransformer

__all__ = ["DecoderOnlyTransformer", "ModelConfig"]
