"""Публичный интерфейс компактной языковой модели."""

from mini_llm.config import ModelConfig
from mini_llm.model import DecoderOnlyTransformer

__all__ = ["DecoderOnlyTransformer", "ModelConfig"]
