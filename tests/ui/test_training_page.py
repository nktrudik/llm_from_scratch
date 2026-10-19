"""Форма обучения не запускает pipeline при открытии или переключении страницы."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mini_llm.ui import client, training_client

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest


@pytest.fixture
def training_app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        client,
        "get_generation_options",
        lambda: client.GenerationOptions(
            model_id="Qwen/Qwen3-0.6B",
            before_sft_available=True,
            after_sft_available=False,
        ),
    )
    monkeypatch.setattr(
        training_client,
        "get_workspace",
        lambda refresh_datasets=True: training_client.WorkspaceOptions(
            datasets=[
                training_client.DatasetOption(
                    dataset_id="2ch",
                    version="v3",
                    title="QA Candidates",
                    samples=1069,
                    splits_ready=True,
                )
            ],
            models=[
                training_client.ModelOption(
                    model_id="Qwen/Qwen3-0.6B", max_sequence_length=512, adaptation_mode="qlora"
                )
            ],
        ),
    )
    return AppTest.from_file(Path(__file__).parents[2] / "src/mini_llm/ui/app.py").run()


def test_training_requires_explicit_form_submission(
    training_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def start(body: dict[str, object]) -> training_client.JobInfo:
        calls.append(body)
        return training_client.JobInfo(job_id="fake", status="queued")

    monkeypatch.setattr(training_client, "start_training", start)
    monkeypatch.setattr(
        training_client,
        "get_job",
        lambda _: training_client.JobInfo(
            job_id="fake",
            status="running",
        ),
    )
    training_app.segmented_control[0].set_value("Обучение").run()
    assert not training_app.exception
    assert not calls
    assert training_app.selectbox[0].value == "2ch/v3"
    next(
        button for button in training_app.button if button.label == "Запустить обучение"
    ).click().run()
    assert not training_app.exception
    assert len(calls) == 1
    assert calls[0]["dataset_version"] == "v3"
    assert calls[0]["model_id"] == "Qwen/Qwen3-0.6B"
    assert "checkpoint_dir" not in calls[0]
    training_app.run()
    assert len(calls) == 1
    training_app.segmented_control[0].set_value("Чат").run()
    assert not training_app.exception
    assert len(calls) == 1


def test_training_displays_api_error_without_crashing(
    training_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> training_client.WorkspaceOptions:
        raise client.GenerationAPIError("Запустите backend")

    monkeypatch.setattr(training_client, "get_workspace", unavailable)
    training_app.segmented_control[0].set_value("Обучение").run()
    assert not training_app.exception
    assert training_app.error[0].value == "Запустите backend"


def test_training_does_not_import_chat_private_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Клиент обучения импортируется даже при старом модуле чата в sys.modules."""

    import importlib
    import json

    import requests

    monkeypatch.delattr(client, "_api_session", raising=False)
    monkeypatch.delattr(client, "_response_payload", raising=False)
    monkeypatch.delattr(client, "api_session")
    monkeypatch.delattr(client, "response_payload")
    reloaded = importlib.reload(training_client)

    def request(*args: object, **kwargs: object) -> requests.Response:
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"datasets": [], "models": [], "runs": []}).encode()
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    assert reloaded.get_workspace().datasets == []


def test_dataset_refresh_fetches_new_versions_without_starting_training(
    training_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def catalog(*, refresh_datasets: bool = True) -> training_client.WorkspaceOptions:
        calls.append(refresh_datasets)
        return training_client.WorkspaceOptions(
            datasets=[
                training_client.DatasetOption(
                    dataset_id="2ch", version="v4", title="Новые QA", samples=500, splits_ready=True
                )
            ],
            models=[
                training_client.ModelOption(
                    model_id="Qwen/Qwen3-0.6B", max_sequence_length=512, adaptation_mode="qlora"
                )
            ],
            dataset_count=1,
            total_samples=500,
        )

    def forbidden(body: dict[str, object]) -> training_client.JobInfo:
        pytest.fail("Обновление каталога не запускает обучение")

    monkeypatch.setattr(training_client, "get_workspace", catalog)
    monkeypatch.setattr(training_client, "start_training", forbidden)
    training_app.segmented_control[0].set_value("Обучение").run()
    assert not training_app.exception
    assert calls == [True]
    next(button for button in training_app.button if button.key == "refresh_datasets").click().run()
    assert not training_app.exception
    assert calls == [True, True]
    assert training_app.selectbox[0].value == "2ch/v4"
    assert "500" in training_app.selectbox[0].options[0]
    training_app.run()
    assert calls == [True, True, True]


def test_ui_shows_live_progress_eta_and_updated_loss(
    training_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mini_llm.observability.training import TrainingProgress

    progress = TrainingProgress(
        status="running",
        stage="Обучение",
        total_steps=100,
        global_step=30,
        epoch=1,
        planned_epochs=2,
        epoch_progress=60,
        eta_seconds=140,
        train_loss=1.25,
        rolling_loss=1.5,
        validation_loss=1.75,
        recent_losses=[1.5, 1.25],
    )
    monkeypatch.setattr(
        training_client,
        "get_job",
        lambda _: training_client.JobInfo(
            job_id="live",
            status="running",
            progress=progress,
        ),
    )
    training_app.session_state["training_job_id"] = "live"
    training_app.segmented_control[0].set_value("Обучение").run(timeout=15)
    assert not training_app.exception
    values = {metric.label: metric.value for metric in training_app.metric}
    assert values["Осталось примерно"] == "≈ 02:20"
    assert values["Train loss"] == "1.2500"
    progress.global_step = 40
    progress.train_loss = 1.0
    progress.eta_seconds = 120
    training_app.run(timeout=15)
    assert not training_app.exception
    values = {metric.label: metric.value for metric in training_app.metric}
    assert values["Train loss"] == "1.0000"
    assert values["Осталось примерно"] == "≈ 02:00"


def test_training_stop_requires_button_and_is_not_repeated(
    training_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = training_client.JobInfo(job_id="active", status="running")
    calls: list[str] = []

    def stop(job_id: str) -> training_client.JobInfo:
        calls.append(job_id)
        job.stop_requested = True
        return job

    monkeypatch.setattr(training_client, "get_job", lambda _: job)
    monkeypatch.setattr(training_client, "stop_training", stop)
    training_app.session_state["training_job_id"] = job.job_id
    training_app.segmented_control[0].set_value("Обучение").run()
    assert not training_app.exception and not calls
    next(
        button for button in training_app.button if button.label == "Остановить обучение"
    ).click().run()
    assert not training_app.exception and calls == ["active"]
    assert next(
        button for button in training_app.button if button.label == "Остановить обучение"
    ).disabled
    training_app.run()
    assert calls == ["active"]
