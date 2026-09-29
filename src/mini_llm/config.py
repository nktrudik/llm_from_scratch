"""Configuration for the decoder-only Transformer."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Hyperparameters and special token IDs for the model architecture."""

    vocab_size: int = 8192
    max_sequence_length: int = 512
    num_layers: int = 4
    d_model: int = 256
    num_heads: int = 4
    ffn_multiplier: int = 4
    dropout: float = 0.1
    bias: bool = True

    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    user_token_id: int = 4
    assistant_token_id: int = 5

    def __post_init__(self) -> None:
        positive_values = {
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ffn_multiplier": self.ffn_multiplier,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        special_token_ids = {
            self.pad_token_id,
            self.unk_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.user_token_id,
            self.assistant_token_id,
        }
        if len(special_token_ids) != 6:
            raise ValueError("Special token IDs must be distinct")
        if min(special_token_ids) < 0 or max(special_token_ids) >= self.vocab_size:
            raise ValueError("Special token IDs must be within the vocabulary")

    @property
    def ffn_size(self) -> int:
        """Return the hidden width of the feed-forward network."""

        return self.ffn_multiplier * self.d_model
