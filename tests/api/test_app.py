"""Быстрые тесты локального FastAPI без запуска тяжёлых pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from mini_llm.api.app import health
from mini_llm.api.jobs import JobAlreadyRunningError, JobManager
from mini_llm.api.schemas import GenerationRequest, TrainingRequest
from mini_llm.main import app


def test_api_registers_control_routes_without_starting_jobs() -> None:
    schema = app.openapi()
    paths = cast(dict[str, object], schema["paths"])

    assert health().status == "ok"
    assert "/v1/training" in paths
    assert "/v1/pretrained/prepare" in paths
    assert "/v1/tokenizer/train" in paths
    assert "/v1/generate" in paths
    assert "/v1/jobs/{job_id}" in paths


def test_job_manager_runs_only_one_heavy_operation() -> None:
    manager = JobManager()
    record = manager.create("training")

    with pytest.raises(JobAlreadyRunningError):
        manager.create("tokenizer_training")

    manager.run(record.job_id, lambda: {"completed": True})
    snapshot = manager.get(record.job_id)
    assert snapshot is not None
    assert snapshot["status"] == "completed"
    assert snapshot["result"] == {"completed": True}
    assert manager.active_job_id() is None


def test_training_request_rejects_two_duration_modes() -> None:
    with pytest.raises(ValidationError, match="ровно один режим"):
        TrainingRequest(epochs=2, max_steps=10)


def test_generation_request_supports_custom_and_pretrained_backends() -> None:
    custom = GenerationRequest(prompt="Привет")
    pretrained = GenerationRequest(
        prompt="Привет",
        model_backend="pretrained",
        pretrained_config_file=Path("configs/pretrained/model.json"),
        tokenizer_file=None,
    )

    assert custom.model_backend == "custom"
    assert custom.to_config().tokenizer_file is not None
    assert pretrained.to_config().model_backend == "pretrained"
    assert pretrained.to_config().tokenizer_file is None


def test_generation_request_validates_backend_specific_files() -> None:
    with pytest.raises(ValidationError, match="pretrained_config_file"):
        GenerationRequest(prompt="Привет", model_backend="pretrained")
    with pytest.raises(ValidationError, match="tokenizer_file"):
        GenerationRequest(prompt="Привет", model_backend="custom", tokenizer_file=None)


def test_generation_swagger_has_separate_backend_examples() -> None:
    schema = GenerationRequest.model_json_schema()
    examples = cast(list[dict[str, object]], schema["examples"])

    assert [example["model_backend"] for example in examples] == ["custom", "pretrained"]
    assert "tokenizer_file" in examples[0]
    assert "tokenizer_file" not in examples[1]
    assert "pretrained_config_file" in examples[1]
