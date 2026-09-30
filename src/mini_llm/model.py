"""PyTorch-модули компактного decoder-only Transformer."""

import math
from typing import cast, overload

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mini_llm.config import ModelConfig


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


class DecoderOnlyTransformer(nn.Module):
    """Компактная decoder-only языковая модель, работающая с token IDs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_sequence_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._initialize_weights)
        self.output_projection.weight = self.token_embeddings.weight
        print(f"DecoderOnlyTransformer parameters: {self.num_parameters():,}")

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        """Вернуть число уникальных параметров модели."""

        parameters = self.parameters()
        if trainable_only:
            return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    @overload
    def forward(self, input_ids: Tensor, targets: None = None) -> Tensor: ...

    @overload
    def forward(self, input_ids: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]: ...

    def forward(
        self, input_ids: Tensor, targets: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Посчитать logits токенов и cross-entropy loss при наличии targets."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence_length]")
        _, sequence_length = input_ids.shape
        if sequence_length == 0:
            raise ValueError("input_ids must contain at least one token")
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds the configured maximum "
                f"of {self.config.max_sequence_length}"
            )

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        hidden_states = cast(Tensor, self.embedding_dropout(hidden_states))
        for module in self.blocks:
            block = cast(TransformerBlock, module)
            hidden_states = block.forward(hidden_states)
        normalized_states = cast(Tensor, self.final_norm(hidden_states))
        logits = cast(Tensor, self.output_projection(normalized_states))

        if targets is None:
            return logits
        if targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")

        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            ignore_index=self.config.pad_token_id,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Авторегрессионно добавить сэмплированные token IDs к prompt."""

        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have shape [batch, sequence_length] with a prompt")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")

        generated = input_ids
        for _ in range(max_new_tokens):
            model_input = generated[:, -self.config.max_sequence_length :]
            next_token_logits = self(model_input)[:, -1, :] / temperature

            if top_k is not None:
                effective_top_k = min(top_k, next_token_logits.size(-1))
                threshold = torch.topk(next_token_logits, effective_top_k).values[:, -1, None]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold, float("-inf")
                )

            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

        return generated
