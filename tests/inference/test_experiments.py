"""Каталог экспериментов, совместимость и безопасность путей без настоящих моделей."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch
from fastapi import HTTPException
from pydantic import ValidationError

from mini_llm.api import app as api_module
from mini_llm.api.schemas import GenerationRequest
from mini_llm.inference import experiments
from mini_llm.inference.config import GenerationConfig
from mini_llm.inference.schemas import GenerationResult
from mini_llm.pretrained import PretrainedConfig
from mini_llm.pretrained.workspace import ACTIVE_MODEL_FILE, ModelPaths, ModelRegistration


@pytest.fixture
def registered_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PretrainedConfig:
    """Подготовить только конфиг и регистрацию, без Hugging Face и весов."""

    monkeypatch.chdir(tmp_path)
    config = PretrainedConfig(
        model_id="Qwen/Qwen3-0.6B", adaptation_mode="qlora", revision="a" * 40
    )
    paths = ModelPaths.for_model(config.model_id)
    config.save(paths.config_file)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    registration = ModelRegistration(
        model_id=config.model_id,
        revision=config.revision,
        config_file=paths.config_file,
        checkpoint_dir=paths.checkpoint_dir,
        snapshot_path=snapshot,
    )
    registration.save(ACTIVE_MODEL_FILE)
    registration.save(paths.registration_file)
    return config


def save_marker(config: PretrainedConfig, experiment_id: str, name: str, step: int) -> Path:
    """Создать крошечный настоящий torch checkpoint для проверки metadata-only чтения."""

    path = experiments.EXPERIMENTS_ROOT / experiment_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_metadata": {
                "backend": "pretrained",
                "pretrained_config": config.model_dump(mode="json"),
            },
            "model_state_dict": {"adapter": torch.zeros(1)},
            "global_step": step,
            "last_validation_loss": 3.4,
        },
        path,
    )
    return path


def test_catalog_filters_incompatible_checkpoints_and_orders_latest_first(
    registered_config: PretrainedConfig,
) -> None:
    older = save_marker(registered_config, "old", "last.pt", 200)
    newer = save_marker(registered_config, "new", "last.pt", 518)
    best = save_marker(registered_config, "new", "best.pt", 400)
    step = save_marker(registered_config, "new", "step_00000300.pt", 300)
    for path in (older, best, step):
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
    for field, value in (
        ("model_id", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("revision", "b" * 40),
        ("lora_rank", 16),
    ):
        save_marker(registered_config.model_copy(update={field: value}), field, "last.pt", 999)
    newer.parent.joinpath("broken.pt").write_bytes(b"invalid checkpoint")
    newer.parent.joinpath("empty.pt").touch()
    newer.parent.joinpath("last.pt.tmp").write_bytes(b"not ready")
    catalog = experiments.list_experiments(registered_config)
    assert [item.experiment_id for item in catalog] == ["new", "old"]
    assert [item.name for item in catalog[0].checkpoints] == [
        "last.pt",
        "best.pt",
        "step_00000300.pt",
    ]
    assert catalog[0].checkpoints[0].global_step == 518


def test_generation_uses_selected_experiment_instead_of_registered_directory(
    registered_config: PretrainedConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = save_marker(registered_config, "new", "last.pt", 518)
    calls: list[GenerationConfig] = []

    def generate(prompt: str, config: GenerationConfig) -> GenerationResult:
        calls.append(config)
        return GenerationResult("Ответ", [1])

    monkeypatch.setattr(api_module, "generate_response", generate)
    request = GenerationRequest(
        prompt="Текст",
        model_backend="pretrained",
        pretrained_model_id=registered_config.model_id,
        pretrained_mode="custom_checkpoint",
        experiment_id="new",
        checkpoint_name="last.pt",
    )
    assert api_module.generate(request).text == "Ответ"
    assert calls[0].checkpoint_file == path
    assert (
        calls[0].pretrained_config_file
        == ModelPaths.for_model(registered_config.model_id).config_file
    )
    options = api_module.generation_options()
    assert options.experiments[0].experiment_id == "new"
    assert options.models[0].experiments[0].checkpoints[0].global_step == 518


def test_wrong_or_missing_experiment_fails_without_model_loading(
    registered_config: PretrainedConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong_config = registered_config.model_copy(update={"lora_rank": 16})
    save_marker(wrong_config, "wrong", "last.pt", 200)

    def unexpected_generation(*args: object, **kwargs: object) -> GenerationResult:
        pytest.fail("Несовместимый checkpoint не должен загружать модель")

    monkeypatch.setattr(api_module, "generate_response", unexpected_generation)
    for experiment_id in ("wrong", "missing"):
        request = GenerationRequest(
            prompt="Текст",
            model_backend="pretrained",
            pretrained_mode="custom_checkpoint",
            experiment_id=experiment_id,
            checkpoint_name="last.pt",
        )
        with pytest.raises(HTTPException) as error:
            api_module.generate(request)
        assert error.value.status_code == 400


@pytest.mark.parametrize("name", ["..", "../outside", "a/b", "a\\b", "C:\\foo", "", "dir:stream"])
def test_experiment_rejects_paths_and_traversal(name: str) -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(
            prompt="Текст",
            model_backend="pretrained",
            pretrained_mode="custom_checkpoint",
            checkpoint_name="last.pt",
            experiment_id=name,
        )


def test_experiment_cannot_escape_via_directory_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_resolve = Path.resolve

    def resolve(path: Path, strict: bool = False) -> Path:
        if path == experiments.EXPERIMENTS_ROOT / "linked":
            return tmp_path / "outside"
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="внутри"):
        experiments.experiment_directory("linked")


def test_experiment_requires_pretrained_checkpoint_mode() -> None:
    with pytest.raises(ValidationError, match="experiment_id"):
        GenerationRequest(prompt="Текст", experiment_id="new")
    with pytest.raises(ValidationError, match="experiment_id"):
        GenerationRequest(
            prompt="Текст",
            model_backend="pretrained",
            pretrained_mode="before_sft",
            experiment_id="new",
        )


def test_metadata_cache_is_invalidated_after_checkpoint_update(
    registered_config: PretrainedConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = save_marker(registered_config, "new", "last.pt", 200)
    original_load = cast(Callable[..., object], torch.load)
    calls = 0

    def load(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        assert kwargs["mmap"] is True and kwargs["weights_only"] is True
        assert kwargs["map_location"] == "cpu"
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    experiments.list_experiments(registered_config)
    experiments.list_experiments(registered_config)
    assert calls == 1
    previous_mtime = path.stat().st_mtime_ns
    save_marker(registered_config, "new", "last.pt", 518)
    os.utime(path, ns=(previous_mtime + 1_000_000, previous_mtime + 1_000_000))
    assert experiments.list_experiments(registered_config)[0].checkpoints[0].global_step == 518
    assert calls == 2


def test_managed_catalog_never_opens_weights_even_on_cold_start(
    registered_config: PretrainedConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mini_llm.workspace.catalog import ModelRun, update_progress

    directory = experiments.EXPERIMENTS_ROOT / "model--qwen--qwen3-0.6b/2ch/v3/run-001"
    directory.mkdir(parents=True)
    config_file = directory / "pretrained.json"
    registered_config.save(config_file)
    run = ModelRun(
        run_id="model--qwen--qwen3-0.6b--2ch-v3--run-001",
        model_id=registered_config.model_id,
        dataset_id="2ch",
        dataset_version="v3",
        directory=directory,
        title="Qwen3 / QA Candidates v3 / run-001",
        global_step=518,
        pretrained_config_file=config_file,
    )
    run.save()
    for name in ("last.pt", "best.pt", "step_00000200.pt"):
        (directory / name).write_bytes(b"not loaded while listing")

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("Каталог managed run не должен вызывать torch.load")

    monkeypatch.setattr(torch, "load", forbidden)
    first = api_module.generation_options().models[0].experiments[0]
    checkpoints = {item.name: item for item in first.checkpoints}
    assert checkpoints["last.pt"].global_step == 518
    assert checkpoints["step_00000200.pt"].global_step == 200
    assert checkpoints["best.pt"].global_step is None
    update_progress(directory, 400, checkpoint_name="best.pt", validation_loss=3.1)
    second = api_module.generation_options().models[0].experiments[0]
    best = next(item for item in second.checkpoints if item.name == "best.pt")
    assert best.global_step == 400
    assert best.validation_loss == 3.1
