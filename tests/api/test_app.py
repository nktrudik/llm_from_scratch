"""Быстрые тесты локального FastAPI без запуска тяжёлых pipeline."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from mini_llm.api.app import health
from mini_llm.api.jobs import JobAlreadyRunningError, JobManager
from mini_llm.api.schemas import TrainingRequest
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
