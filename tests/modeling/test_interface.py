"""Тест общего model interface на собственной Transformer-модели."""

import torch
from torch import Tensor, nn

from mini_llm.modeling import CustomCausalLMBackend, DecoderOnlyTransformer, ModelConfig
from mini_llm.pretrained import HuggingFaceCausalLMBackend, PretrainedConfig


class _FakeOutput:
    def __init__(self, logits: Tensor, loss: Tensor) -> None:
        self.logits = logits
        self.loss = loss


class _FakeHuggingFaceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 8)
        self.received_labels: Tensor | None = None

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        use_cache: bool,
    ) -> _FakeOutput:
        del attention_mask, use_cache
        self.received_labels = labels
        logits = self.projection(torch.nn.functional.one_hot(input_ids, num_classes=4).float())
        return _FakeOutput(logits, logits.sum() * 0 + 1.0)


def test_custom_model_works_through_common_backend() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_sequence_length=8,
        num_layers=1,
        d_model=16,
        num_heads=4,
        dropout=0.0,
    )
    backend = CustomCausalLMBackend(DecoderOnlyTransformer(config), config)
    input_ids = torch.randint(6, config.vocab_size, (2, 5))
    labels = torch.randint(6, config.vocab_size, (2, 5))
    labels[:, :3] = -100

    output = backend.forward_batch(input_ids, torch.ones_like(input_ids), labels)
    expected_loss = torch.nn.functional.cross_entropy(
        output.logits.reshape(-1, config.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )

    assert output.logits.shape == (2, 5, config.vocab_size)
    assert torch.isfinite(output.loss)
    torch.testing.assert_close(output.loss, expected_loss)
    assert backend.checkpoint_metadata["backend"] == "custom"


def test_pretrained_model_uses_the_same_backend_contract() -> None:
    module = _FakeHuggingFaceModel()
    config = PretrainedConfig(model_id="org/model", max_sequence_length=16)
    backend = HuggingFaceCausalLMBackend(module, config)
    input_ids = torch.randint(0, 4, (1, 5))
    labels = torch.tensor([[-100, -100, 1, 2, 3]])

    output = backend.forward_batch(input_ids, torch.ones_like(input_ids), labels)

    assert output.logits.shape == (1, 5, 8)
    assert module.received_labels is labels
    assert backend.checkpoint_metadata["backend"] == "pretrained"
