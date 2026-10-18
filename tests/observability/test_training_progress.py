"""Детерминированные тесты ETA, resume и публикации метрик без обучения."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_llm.api import app as api_module
from mini_llm.api.jobs import JobManager
from mini_llm.observability import terminal_stage
from mini_llm.observability import training as live
from mini_llm.workspace.catalog import ModelRun


def test_eta_excludes_preparation_and_previous_resume_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr("mini_llm.observability.training.time.perf_counter", lambda: clock[0])
    reporter = live.TrainingProgressReporter(tmp_path)
    reporter.snapshot.total_steps = 210
    clock[0] = 100
    reporter.start(step=200, tokens=1000)
    assert reporter.snapshot.eta_seconds is None
    clock[0] = 112
    reporter.snapshot.global_step = 203
    reporter.snapshot.tokens_seen = 1120
    reporter.publish()
    assert reporter.snapshot.eta_seconds == pytest.approx(28)
    assert reporter.snapshot.tokens_per_second == pytest.approx(10)
    reporter.finish()
    saved = live.read_training_progress(tmp_path)
    assert saved is not None
    assert saved.status == "completed"
    assert saved.eta_seconds == 0


def test_stage_progress_and_failures_are_visible_and_context_is_reset(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="synthetic"), live.report_training(tmp_path) as reporter:
        reporter.start(step=0, tokens=0)
        with terminal_stage("VALIDATION", "Проверка validation"):
            saved = live.read_training_progress(tmp_path)
            assert saved is not None and saved.stage == "Проверка validation"
        assert reporter.snapshot.stage == "Обучение"
        raise RuntimeError("synthetic")
    final = live.read_training_progress(tmp_path)
    assert final is not None and final.status == "failed"
    assert final.error == "synthetic"
    assert live.current_training_progress() is None


def test_writes_are_throttled_and_invalid_loss_does_not_break_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr("mini_llm.observability.training.time.perf_counter", lambda: clock[0])
    reporter = live.TrainingProgressReporter(tmp_path)
    reporter.snapshot.global_step = 1
    clock[0] = 0.1
    reporter.publish()
    saved = live.read_training_progress(tmp_path)
    assert saved is not None and saved.global_step == 0
    clock[0] = 1.1
    reporter.snapshot.train_loss = float("nan")
    reporter.snapshot.recent_losses = [float("nan"), 1.0]
    reporter.publish()
    saved = live.read_training_progress(tmp_path)
    assert saved is not None and saved.global_step == 1
    assert saved.train_loss is None and saved.recent_losses == [1.0]


def test_job_api_returns_live_steps_not_saved_checkpoint_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    manager = JobManager()
    job = manager.create("training")
    monkeypatch.setattr(api_module, "job_manager", manager)
    directory = Path("checkpoints/runs/custom/test/v1/run-001")
    ModelRun(
        run_id="custom-test-v1-run-001",
        model_id="custom",
        dataset_id="test",
        dataset_version="v1",
        directory=directory,
        title="Тест",
        job_id=job.job_id,
        global_step=0,
    ).save()
    reporter = live.TrainingProgressReporter(directory)
    reporter.start(step=0, tokens=0)
    reporter.snapshot.global_step = 3
    reporter.snapshot.total_steps = 10
    reporter.publish(force=True)
    response = api_module.get_job(job.job_id)
    assert response.progress is not None
    assert response.progress.global_step == 3
    assert response.progress.total_steps == 10
    assert not list(directory.glob("*.pt"))
