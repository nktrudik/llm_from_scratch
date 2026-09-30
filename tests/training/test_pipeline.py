"""Быстрые тесты training config и checkpoint resume."""

from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from mini_llm.modeling import CustomCausalLMBackend, DecoderOnlyTransformer, ModelConfig
from mini_llm.training import TrainingConfig
from mini_llm.training.checkpoints import load_checkpoint, save_checkpoint
from mini_llm.training.overfit import OverfitConfig
from mini_llm.training.progress import calculate_training_plan, format_training_progress
from mini_llm.training.schemas import TrainingState


def _small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        max_sequence_length=16,
        num_layers=1,
        d_model=32,
        num_heads=4,
        dropout=0.0,
    )


def test_training_batch_size_uses_shared_limit() -> None:
    assert TrainingConfig().batch_size == 8
    assert TrainingConfig().validation_interval == 1000
    assert TrainingConfig().validation_batches >= 200
    assert TrainingConfig(batch_size=1).batch_size == 1
    assert TrainingConfig(batch_size=8).batch_size == 8
    with pytest.raises(ValueError, match="от 1 до 8"):
        TrainingConfig(batch_size=9)
    with pytest.raises(ValueError, match="от 1 до 4"):
        OverfitConfig(batch_size=5)


def test_training_plan_uses_complete_epochs_or_explicit_step_mode() -> None:
    epoch_plan = calculate_training_plan(10, 4, epochs=3, max_steps=None)
    assert epoch_plan.steps_per_epoch == 3
    assert epoch_plan.planned_total_steps == 9
    assert epoch_plan.planned_epochs == 3

    step_plan = calculate_training_plan(10, 4, epochs=None, max_steps=7)
    assert step_plan.steps_per_epoch == 3
    assert step_plan.planned_total_steps == 7
    assert step_plan.planned_epochs == 3

    with pytest.raises(ValueError, match="ровно один режим"):
        TrainingConfig(epochs=3, max_steps=7)


def test_training_progress_contains_seen_counts_and_epoch_percentage() -> None:
    line = format_training_progress(
        epoch=2,
        planned_epochs=3,
        epoch_progress=25.0,
        global_step=125,
        planned_total_steps=300,
        samples_seen=500,
        tokens_seen=123_456,
        train_loss=2.5,
        rolling_loss=2.75,
        learning_rate=3e-4,
        tokens_per_second=1234.5,
        telemetry="device=cpu",
    )
    assert "epoch_progress=25.00%" in line
    assert "samples_seen=500" in line
    assert "tokens_seen=123456" in line
    assert "rolling_train_loss_100=2.750000" in line


def test_checkpoint_restores_model_optimizer_and_training_state(tmp_path: Path) -> None:
    config = _small_config()
    model = DecoderOnlyTransformer(config)
    backend = CustomCausalLMBackend(model, config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    input_ids = torch.randint(6, config.vocab_size, (2, 8))
    targets = torch.randint(6, config.vocab_size, (2, 8))
    _, loss = model(input_ids, targets)
    loss.backward()
    optimizer.step()
    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    state = TrainingState(
        epoch=2,
        batches_completed_in_epoch=7,
        global_step=19,
        best_validation_loss=1.25,
        last_train_loss=1.5,
        last_validation_loss=1.25,
        samples_seen=73,
        tokens_seen=4096,
        recent_train_losses=[1.8, 1.5],
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=backend,
        optimizer=optimizer,
        scaler=scaler,
        training_config={"batch_size": 1},
        state=state,
    )

    restored_model = DecoderOnlyTransformer(config)
    restored_backend = CustomCausalLMBackend(restored_model, config)
    restored_optimizer = AdamW(restored_model.parameters(), lr=9e-4)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored_state = load_checkpoint(
        checkpoint_path,
        model=restored_backend,
        optimizer=restored_optimizer,
        scaler=restored_scaler,
        learning_rate=5e-4,
        expected_batch_size=1,
    )

    assert restored_state == state
    assert restored_optimizer.param_groups[0]["lr"] == 5e-4
    for expected, actual in zip(expected_parameters, restored_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
