"""Доступность режимов UI и выбор best checkpoint из активной регистрации."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_llm.api.schemas import GenerationRequest
from mini_llm.inference.options import pretrained_generation_options
from mini_llm.pretrained import PretrainedConfig
from mini_llm.pretrained.workspace import ACTIVE_MODEL_FILE, ModelRegistration


@pytest.fixture
def registered_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelRegistration:
    """Создать конфиг и регистрацию; веса и checkpoints не создавать."""

    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.json"
    PretrainedConfig(model_id="org/instruct", revision="a" * 40).save(config_file)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    registration = ModelRegistration(
        model_id="org/instruct",
        revision="a" * 40,
        config_file=config_file,
        checkpoint_dir=tmp_path / "checkpoints",
        snapshot_path=snapshot,
    )
    registration.checkpoint_dir.mkdir()
    registration.save(ACTIVE_MODEL_FILE)
    return registration


def test_unprepared_model_disables_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    options = pretrained_generation_options()
    assert not options.before_sft_available
    assert not options.after_sft_available
    assert options.reason is not None


def test_after_sft_requires_best_not_periodic_checkpoint(
    registered_model: ModelRegistration,
) -> None:
    registered_model.checkpoint_dir.joinpath("step_00001000.pt").write_bytes(b"periodic")
    options = pretrained_generation_options()
    assert options.model_id == "org/instruct"
    assert options.before_sft_available
    assert not options.after_sft_available
    registered_model.best_checkpoint.write_bytes(b"best")
    assert pretrained_generation_options().after_sft_available


def test_empty_best_is_not_available(registered_model: ModelRegistration) -> None:
    registered_model.best_checkpoint.touch()
    assert not pretrained_generation_options().after_sft_available


def test_pretrained_request_uses_active_model_and_selected_mode(
    registered_model: ModelRegistration,
) -> None:
    before = GenerationRequest(prompt="text", model_backend="pretrained").to_config()
    after = GenerationRequest(
        prompt="text", model_backend="pretrained", pretrained_mode="after_sft"
    ).to_config()
    assert before.pretrained_config_file == registered_model.config_file
    assert before.checkpoint_file is None
    assert before.pretrained_mode == "before_sft"
    assert after.pretrained_config_file == registered_model.config_file
    assert after.checkpoint_file == registered_model.best_checkpoint
    assert after.pretrained_mode == "after_sft"


def test_mismatched_registration_disables_generation(registered_model: ModelRegistration) -> None:
    PretrainedConfig(model_id="different/model", revision="a" * 40).save(
        registered_model.config_file
    )
    assert not pretrained_generation_options().before_sft_available


def test_custom_rejects_pretrained_mode() -> None:
    with pytest.raises(ValidationError, match="только для pretrained"):
        GenerationRequest(prompt="text", model_backend="custom", pretrained_mode="before_sft")


def test_before_sft_rejects_explicit_checkpoint() -> None:
    with pytest.raises(ValidationError, match="не использует checkpoint"):
        GenerationRequest(
            prompt="text",
            model_backend="pretrained",
            pretrained_mode="before_sft",
            checkpoint_file=Path("best.pt"),
            pretrained_config_file=Path("config.json"),
        )
