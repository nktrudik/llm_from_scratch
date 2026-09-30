"""MLP и residual-блоки Transformer."""

from typing import cast

from torch import Tensor, nn

from mini_llm.modeling.attention import CausalSelfAttention
from mini_llm.modeling.config import ModelConfig


class MLP(nn.Module):
    """Позиционная feed-forward сеть каждого Transformer-блока."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_size, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.ffn_size, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.layers(inputs))


class TransformerBlock(nn.Module):
    """Decoder-блок с pre-norm, attention и остаточными путями MLP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, inputs: Tensor) -> Tensor:
        attention_inputs = cast(Tensor, self.attention_norm(inputs))
        inputs = inputs + self.attention.forward(attention_inputs)
        mlp_inputs = cast(Tensor, self.mlp_norm(inputs))
        return inputs + self.mlp.forward(mlp_inputs)
