"""Tests for the decoder-only Transformer architecture."""

import torch

from mini_llm import DecoderOnlyTransformer, ModelConfig


def small_config() -> ModelConfig:
    """Return a fast configuration for behavioral tests."""

    return ModelConfig(
        vocab_size=64,
        max_sequence_length=16,
        num_layers=2,
        d_model=32,
        num_heads=4,
        dropout=0.0,
    )


def test_model_is_created_on_cpu() -> None:
    model = DecoderOnlyTransformer(small_config())

    assert next(model.parameters()).device.type == "cpu"


def test_logits_have_expected_shape() -> None:
    config = small_config()
    model = DecoderOnlyTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 7))

    logits = model(input_ids)

    assert logits.shape == (2, 7, config.vocab_size)


def test_targets_return_finite_loss() -> None:
    config = small_config()
    model = DecoderOnlyTransformer(config)
    input_ids = torch.randint(4, config.vocab_size, (2, 7))
    targets = torch.randint(4, config.vocab_size, (2, 7))

    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 7, config.vocab_size)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_future_tokens_do_not_affect_earlier_logits() -> None:
    config = small_config()
    model = DecoderOnlyTransformer(config).eval()
    original = torch.randint(0, config.vocab_size, (1, 8))
    changed = original.clone()
    position = 3
    changed[:, position + 1 :] = (changed[:, position + 1 :] + 1) % config.vocab_size

    original_logits = model(original)
    changed_logits = model(changed)

    torch.testing.assert_close(
        original_logits[:, : position + 1],
        changed_logits[:, : position + 1],
        rtol=0.0,
        atol=0.0,
    )


def test_generate_appends_requested_number_of_token_ids() -> None:
    config = small_config()
    model = DecoderOnlyTransformer(config).eval()
    prompt = torch.tensor([[config.bos_token_id, 7, 12]], dtype=torch.long)

    generated = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=10)

    assert generated.shape == (1, prompt.size(1) + 5)
    assert torch.equal(generated[:, : prompt.size(1)], prompt)


def test_default_parameter_count_is_in_expected_range() -> None:
    model = DecoderOnlyTransformer(ModelConfig())

    assert 5_000_000 <= model.num_parameters() <= 5_500_000
    assert model.output_projection.weight is model.token_embeddings.weight
