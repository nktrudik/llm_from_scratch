"""Причинный multi-head self-attention."""

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mini_llm.modeling.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention с фиксированной причинной маской видимости."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.d_model // config.num_heads

        self.qkv_projection = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.output_projection = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(
            torch.ones(config.max_sequence_length, config.max_sequence_length, dtype=torch.bool)
        )
        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, config.max_sequence_length, config.max_sequence_length),
            persistent=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, sequence_length, embedding_size = inputs.shape
        queries, keys, values = self.qkv_projection(inputs).chunk(3, dim=-1)

        queries = queries.view(
            batch_size, sequence_length, self.num_heads, self.head_size
        ).transpose(1, 2)
        keys = keys.view(batch_size, sequence_length, self.num_heads, self.head_size).transpose(
            1, 2
        )
        values = values.view(batch_size, sequence_length, self.num_heads, self.head_size).transpose(
            1, 2
        )

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        causal_mask = cast(Tensor, self.causal_mask)
        visible_positions = causal_mask[:, :, :sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(~visible_positions, float("-inf"))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attended_values = attention_weights @ values
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, sequence_length, embedding_size)
        return cast(Tensor, self.residual_dropout(self.output_projection(attended_values)))
