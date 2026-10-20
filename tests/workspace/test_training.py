"""Управляемый запуск проверяется с подменой trainer, без реального обучения."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError

from mini_llm.api import app as api_module
from mini_llm.api.jobs import JobManager
from mini_llm.training.config import TrainingConfig
from mini_llm.training.schemas import TrainingResult
from mini_llm.workspace import training
from mini_llm.workspace.catalog import list_runs
from mini_llm.workspace.datasets import publish_dataset
from mini_llm.workspace.training import ManagedTrainingRequest, TrainingParameters


@pytest.fixture
def managed_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ManagedTrainingRequest:
    monkeypatch.chdir(tmp_path)
    rows = [
        json.dumps(
            {
                "board": "b",
                "thread_id": n,
                "context": [{"text": f"Вопрос {n}"}],
                "response": {"text": f"Ответ {n}"},
            }
        )
        + "\n"
        for n in range(3)
    ]
    source = tmp_path / "samples.jsonl"
    source.write_text("".join(rows), encoding="utf-8")
    dataset = publish_dataset(source, "2ch", "v1", "Тестовые пары")
    directory = dataset.directory / "splits"
    directory.mkdir()
    for name, row in zip(("train", "validation", "test"), rows, strict=True):
        (directory / f"{name}.jsonl").write_text(row, encoding="utf-8")
    return ManagedTrainingRequest(
        dataset_id="2ch",
        dataset_version="v1",
        model_id="custom",
        parameters=TrainingParameters(max_sequence_length=1024, device="cpu"),
    )


def test_managed_training_and_resume_always_use_new_run(
    managed_request: ManagedTrainingRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[TrainingConfig] = []

    def fake_train(config: TrainingConfig) -> TrainingResult:
        calls.append(config)
        path = config.checkpoint_dir / "last.pt"
        path.write_bytes(b"immutable checkpoint")
        return TrainingResult(2, 1.0, path, False)

    monkeypatch.setattr(training, "train_model", fake_train)
    result = training.run_managed_training(managed_request, "job-1")
    first = list_runs()[0]
    assert result["global_step"] == 2
    assert first.status == "completed"
    assert first.training_config["dataset_version"] == "v1"
    assert "split_sha256" in first.training_config
    resumed = managed_request.model_copy(update={"parent_run_id": first.run_id})
    training.run_managed_training(resumed, "job-2")
    assert len(list_runs()) == 2
    assert calls[0].checkpoint_dir != calls[1].checkpoint_dir
    assert calls[1].resume_from == calls[0].checkpoint_dir / "last.pt"
    assert calls[0].splits_dir == Path("data/datasets/2ch/v1/splits")
    assert (first.directory / "last.pt").read_bytes() == b"immutable checkpoint"


def test_failed_run_remains_visible_with_error(
    managed_request: ManagedTrainingRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failure(config: TrainingConfig) -> TrainingResult:
        raise RuntimeError("Недостаточно памяти")

    monkeypatch.setattr(training, "train_model", failure)
    with pytest.raises(RuntimeError, match="памяти"):
        training.run_managed_training(managed_request, "job")
    assert list_runs()[0].status == "failed"
    assert "памяти" in str(list_runs()[0].error)


def test_invalid_duration_and_other_dataset_are_rejected(
    managed_request: ManagedTrainingRequest,
) -> None:
    with pytest.raises(ValidationError):
        TrainingParameters(epochs=1, max_steps=3)
    invalid = managed_request.model_copy(update={"parent_run_id": "unknown"})
    with pytest.raises(ValueError, match="ту же версию"):
        training.validate_training_request(invalid)


def test_api_only_queues_training_after_explicit_request(
    managed_request: ManagedTrainingRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_module, "job_manager", JobManager())
    tasks = BackgroundTasks()
    response = api_module.start_versioned_training(managed_request, tasks)
    assert response.status == "queued"
    assert len(tasks.tasks) == 1
    assert not list_runs()
    with pytest.raises(HTTPException) as error:
        api_module.start_versioned_training(managed_request, BackgroundTasks())
    assert error.value.status_code == 409


def test_pretrained_config_is_copied_per_run_without_changing_qlora(
    managed_request: ManagedTrainingRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mini_llm.pretrained.config import PretrainedConfig
    from mini_llm.pretrained.workspace import ModelPaths, ModelRegistration

    model_id = "Qwen/Qwen3-0.6B"
    paths = ModelPaths.for_model(model_id)
    base = PretrainedConfig(model_id=model_id, adaptation_mode="qlora")
    base.save(paths.config_file)
    ModelRegistration(
        model_id=model_id,
        revision=base.revision,
        config_file=paths.config_file,
        checkpoint_dir=paths.checkpoint_dir,
        snapshot_path=Path(".cache/test"),
    ).save(paths.registration_file)
    calls: list[TrainingConfig] = []

    def fake_train(config: TrainingConfig) -> TrainingResult:
        calls.append(config)
        path = config.checkpoint_dir / "last.pt"
        path.write_bytes(b"no weights loaded")
        return TrainingResult(1, 1.0, path, False)

    monkeypatch.setattr(training, "train_model", fake_train)
    selected = managed_request.model_copy(
        update={
            "model_id": model_id,
            "parameters": TrainingParameters(device="cpu", num_workers=0),
        }
    )
    training.run_managed_training(selected, "fake")
    run = list_runs()[0]
    assert calls[0].pretrained_config_file == run.directory / "pretrained.json"
    assert calls[0].gradient_accumulation_steps == 2
    assert calls[0].lr_scheduler_type == "cosine" and calls[0].warmup_steps == 30
    assert calls[0].packing
    assert PretrainedConfig.load(calls[0].pretrained_config_file) == base
    assert PretrainedConfig.load(paths.config_file) == base
    assert run.tokenizer_file is None
    assert run.dataset_version == "v1"


def test_training_cli_without_output_uses_versioned_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mini_llm.training import cli

    calls: list[TrainingConfig] = []

    def fake(config: TrainingConfig) -> TrainingResult:
        calls.append(config)
        return TrainingResult(0, 1.0, Path("last.pt"), False)

    monkeypatch.setattr(training, "train_versioned_config", fake)
    assert cli.main(["--epochs", "1"]) == 0
    assert calls[0].splits_dir == Path("data/datasets/2ch/v3/splits")
