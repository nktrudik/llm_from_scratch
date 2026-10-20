"""API очереди checkpoints использует тот же GPU-слот и не принимает пути отчётов."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from fastapi import BackgroundTasks, HTTPException
from fastapi.routing import APIRoute

from mini_llm.api import evaluations
from mini_llm.api.jobs import JobAlreadyRunningError, JobManager
from mini_llm.api.schemas import JobResponse
from mini_llm.inference.evaluation.schemas import EvaluationProgress, EvaluationRequest
from mini_llm.workspace.catalog import ModelRun


def test_evaluation_uses_exclusive_job_slot_progress_and_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager()
    run = ModelRun(
        run_id="safe-run",
        model_id="custom",
        dataset_id="2ch",
        dataset_version="v5",
        directory=Path("unused"),
        title="test",
    )
    monkeypatch.setattr(evaluations, "evaluation_plan", lambda _: (run, [Path("last.pt")]))
    calls: list[str] = []

    def execute(
        request: EvaluationRequest,
        *,
        report_progress: Callable[[EvaluationProgress], None],
    ) -> dict[str, object]:
        calls.append(request.run_id)
        snapshot = EvaluationProgress(
            total=5, total_checkpoints=1, report_file="artifacts/evaluations/test.md"
        )
        report_progress(snapshot)
        assert manager.get(record.job_id) is not None
        with pytest.raises(JobAlreadyRunningError), manager.generation():
            pytest.fail("Генерация не должна пересекаться с прогоном")
        return {"completed": 5}

    monkeypatch.setattr(evaluations, "run_evaluation", execute)
    router = evaluations.create_evaluation_router(manager)
    endpoint = next(
        r.endpoint
        for r in router.routes
        if isinstance(r, APIRoute)
        and r.path == "/v1/evaluations"
        and r.methods is not None
        and "POST" in r.methods
    )
    start = cast(Callable[[EvaluationRequest, BackgroundTasks], JobResponse], endpoint)
    tasks = BackgroundTasks()
    record = start(EvaluationRequest(run_id="safe-run"), tasks)
    assert not calls and record.status == "queued"
    with pytest.raises(HTTPException) as error:
        start(EvaluationRequest(run_id="safe-run"), BackgroundTasks())
    assert error.value.status_code == 409
    with pytest.raises(JobAlreadyRunningError):
        manager.create("training")
    assert manager.request_stop(record.job_id)["stop_requested"]
    task = tasks.tasks[0]
    task.func(*task.args, **task.kwargs)
    snapshot = manager.get(record.job_id)
    assert snapshot is not None and snapshot["status"] == "completed"
    assert calls == ["safe-run"] and snapshot["evaluation_progress"] is not None
    assert manager.active_job_id() is None


def test_report_download_rejects_unknown_jobs_and_paths_outside_reports() -> None:
    manager = JobManager()
    router = evaluations.create_evaluation_router(manager)
    endpoint = next(
        r.endpoint
        for r in router.routes
        if isinstance(r, APIRoute) and r.path == "/v1/evaluations/{job_id}/report"
    )
    report = cast(Callable[[str], object], endpoint)
    with pytest.raises(HTTPException) as error:
        report("missing")
    assert error.value.status_code == 404
    record = manager.create("checkpoint_evaluation")
    manager.report_evaluation(record.job_id, {"report_file": "outside-private-file.md"})
    with pytest.raises(HTTPException) as error:
        report(record.job_id)
    assert error.value.status_code == 404
