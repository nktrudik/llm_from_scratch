"""Настоящий Streamlit-экран проверки с заглушкой API, без inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mini_llm.inference.evaluation.schemas import (
    QUESTIONS,
    EvaluationAnswer,
    EvaluationJob,
    EvaluationProgress,
)
from mini_llm.ui import client, evaluation_client, training_client
from mini_llm.workspace.catalog import CheckpointMetadata, ModelRun

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest


@pytest.fixture
def evaluation_app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    run = ModelRun(
        run_id="model--qwen--qwen3--2ch-v5--run-001",
        model_id="Qwen/Qwen3-0.6B",
        dataset_id="2ch",
        dataset_version="v5",
        title="Новый запуск",
        directory=Path("checkpoints/runs/model/2ch/v5/run-001"),
        checkpoints={"last.pt": CheckpointMetadata(global_step=50, modified_ns=0, size=1)},
    )
    monkeypatch.setattr(
        client,
        "get_generation_options",
        lambda: client.GenerationOptions(
            model_id=run.model_id,
            before_sft_available=False,
            after_sft_available=False,
        ),
    )
    monkeypatch.setattr(
        training_client,
        "get_workspace",
        lambda **_: training_client.WorkspaceOptions(
            datasets=[],
            models=[],
            runs=[run],
        ),
    )
    monkeypatch.setattr(
        evaluation_client,
        "get_plan",
        lambda _: evaluation_client.EvaluationPlan(
            run_id=run.run_id,
            checkpoints=["step_00000050.pt", "last.pt"],
            questions=list(QUESTIONS),
            total=10,
        ),
    )
    return AppTest.from_file(Path(__file__).parents[2] / "src/mini_llm/ui/app.py").run()


def test_evaluation_starts_only_on_click_and_shows_one_latest_answer(
    evaluation_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    job = EvaluationJob(
        job_id="fake",
        kind="checkpoint_evaluation",
        status="running",
        evaluation_progress=EvaluationProgress(
            total=10,
            total_checkpoints=2,
            completed=2,
            report_file="artifacts/evaluations/test.md",
            eta_seconds=80,
            latest_result=EvaluationAnswer(
                checkpoint="last.pt",
                question_number=2,
                question=QUESTIONS[1],
                answer="Первый ответ",
                generated_tokens=2,
                elapsed_seconds=1,
            ),
        ),
    )

    def start(run_id: str) -> EvaluationJob:
        calls.append(run_id)
        return job

    monkeypatch.setattr(evaluation_client, "start_evaluation", start)
    monkeypatch.setattr(evaluation_client, "get_job", lambda _: job)
    evaluation_app.segmented_control[0].set_value("Проверка checkpoints").run()
    assert not evaluation_app.exception and not calls
    next(b for b in evaluation_app.button if b.label == "Проверить все checkpoints").click().run()
    assert not evaluation_app.exception and len(calls) == 1
    assert len(evaluation_app.chat_message) == 2
    assert any(t.value == "Первый ответ" for t in evaluation_app.text)
    assert job.evaluation_progress is not None and job.evaluation_progress.latest_result is not None
    job.evaluation_progress.latest_result.answer = "Следующий ответ"
    job.evaluation_progress.completed = 3
    evaluation_app.run()
    assert not evaluation_app.exception and len(calls) == 1
    assert len(evaluation_app.chat_message) == 2
    assert not any(t.value == "Первый ответ" for t in evaluation_app.text)
    assert any(t.value == "Следующий ответ" for t in evaluation_app.text)


def test_evaluation_api_failure_does_not_crash_ui(
    evaluation_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mini_llm.ui.http import APIError

    def unavailable(run_id: str) -> evaluation_client.EvaluationPlan:
        raise APIError("API недоступен")

    monkeypatch.setattr(evaluation_client, "get_plan", unavailable)
    evaluation_app.segmented_control[0].set_value("Проверка checkpoints").run()
    assert not evaluation_app.exception and evaluation_app.error
