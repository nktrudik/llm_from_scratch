"""Быстрые тесты локального FastAPI без запуска тяжёлых pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from fastapi import BackgroundTasks
from pydantic import ValidationError

from mini_llm.api.app import health
from mini_llm.api.jobs import JobAlreadyRunningError, JobManager
from mini_llm.api.schemas import GenerationRequest, TrainingRequest
from mini_llm.inference import GenerationConfig
from mini_llm.inference.config import DEFAULT_PRETRAINED_CONFIG_PATH
from mini_llm.inference.schemas import GenerationResult
from mini_llm.main import app
from mini_llm.ui.config import ModelBackend


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
        GenerationRequest(prompt="Привет", model_backend="pretrained", pretrained_config_file=None)
    with pytest.raises(ValidationError, match="tokenizer_file"):
        GenerationRequest(prompt="Привет", model_backend="custom", tokenizer_file=None)


def test_generation_swagger_has_separate_backend_examples() -> None:
    schema = GenerationRequest.model_json_schema()
    examples = cast(list[dict[str, object]], schema["examples"])

    assert [example["model_backend"] for example in examples] == ["custom", "pretrained"]
    assert "tokenizer_file" in examples[0]
    assert "tokenizer_file" not in examples[1]
    assert "pretrained_config_file" in examples[1]


def test_generation_defaults_depend_on_backend() -> None:
    custom = GenerationRequest(prompt="Привет", model_backend="custom").to_config()
    pretrained = GenerationRequest(prompt="Привет", model_backend="pretrained").to_config()

    assert custom.checkpoint_file == Path("checkpoints/training/best.pt")
    assert custom.tokenizer_file == Path("artifacts/tokenizer/2ch_bpe.json")
    assert custom.pretrained_config_file is None
    assert pretrained.checkpoint_file is None
    assert pretrained.pretrained_mode == "before_sft"
    assert pretrained.pretrained_config_file == DEFAULT_PRETRAINED_CONFIG_PATH
    assert pretrained.tokenizer_file is None
    assert pretrained.device == custom.device == "cuda"
    assert pretrained.max_new_tokens == custom.max_new_tokens == 256


def test_generation_pretrained_explicit_paths_override_defaults() -> None:
    request = GenerationRequest(
        prompt="Привет",
        model_backend="pretrained",
        checkpoint_file=Path("another/last.pt"),
        pretrained_config_file=Path("another/model.json"),
    )

    assert request.to_config().checkpoint_file == Path("another/last.pt")
    assert request.to_config().pretrained_config_file == Path("another/model.json")


@pytest.mark.parametrize("backend", ["custom", "pretrained"])
def test_generation_route_accepts_minimal_ui_payload(
    monkeypatch: pytest.MonkeyPatch, backend: ModelBackend
) -> None:
    from mini_llm.api import app as api_module

    configs: list[GenerationConfig] = []

    def generate_response(prompt: str, config: GenerationConfig) -> GenerationResult:
        assert prompt == "Привет"
        configs.append(config)
        return GenerationResult(text="Ответ", token_ids=[10])

    monkeypatch.setattr(api_module, "generate_response", generate_response)
    response = api_module.generate(
        GenerationRequest.model_validate({"prompt": "Привет", "model_backend": backend})
    )

    assert response.text == "Ответ"
    assert configs[0].model_backend == backend


@pytest.mark.parametrize(
    ("kind", "should_clear"),
    [("training", True), ("pretrained_prepare", True), ("scraper", False)],
)
def test_gpu_jobs_release_inference_cache(
    monkeypatch: pytest.MonkeyPatch, kind: str, should_clear: bool
) -> None:
    from mini_llm.api import app as api_module

    calls: list[str] = []
    monkeypatch.setattr(api_module, "job_manager", JobManager())
    monkeypatch.setattr(api_module, "clear_pretrained_cache", lambda: calls.append("clear"))
    result = api_module._submit_job(kind, BackgroundTasks(), lambda: None)

    assert result.status == "queued"
    assert calls == (["clear"] if should_clear else [])
