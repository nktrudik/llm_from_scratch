"""Вкладка моделей: никакой автоматической загрузки или очистки при открытии."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mini_llm.pretrained.management.schemas import (
    DeletionPath,
    DeletionPreview,
    DeletionRequest,
    DeletionTarget,
    ManagedModel,
    ManagedRun,
    ModelsCatalog,
)
from mini_llm.ui import client, models_client

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest


def job(status: str = "running") -> models_client.ModelJob:
    return models_client.ModelJob.model_validate(
        {
            "job_id": "abc123",
            "kind": "model_download",
            "status": status,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )


@pytest.fixture
def models_app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        client,
        "get_generation_options",
        lambda: client.GenerationOptions(
            model_id="org/model",
            before_sft_available=True,
            after_sft_available=False,
        ),
    )
    catalog = ModelsCatalog(
        models=[
            ManagedModel(
                model_id="org/model",
                registered=True,
                available=True,
                size_bytes=2048,
                runs=[
                    ManagedRun(
                        run_id="test-run",
                        title="Модель / v1 / run-001",
                        model_id="org/model",
                        directory="checkpoints/runs/test",
                        checkpoints=["best.pt", "last.pt"],
                        size_bytes=1024,
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(models_client, "get_models", lambda: catalog)
    monkeypatch.setattr(models_client, "get_model_job", lambda _: job())
    return AppTest.from_file(Path(__file__).parents[2] / "src/mini_llm/ui/app.py").run()


def test_model_page_only_downloads_after_form_submission(
    models_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def download(model_id: str) -> models_client.ModelJob:
        calls.append(model_id)
        return job()

    monkeypatch.setattr(models_client, "download_model", download)
    models_app.segmented_control[0].set_value("Модели").run()
    assert not models_app.exception
    assert not calls
    models_app.text_input[0].set_value("org/new")
    next(button for button in models_app.button if button.label == "Загрузить модель").click().run()
    assert not models_app.exception
    assert calls == ["org/new"]
    models_app.run()
    assert calls == ["org/new"]


def test_delete_requires_preview_and_exact_confirmation(
    models_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[DeletionRequest] = []

    def preview(target: DeletionTarget) -> DeletionPreview:
        return DeletionPreview(
            target=target,
            paths=[DeletionPath(path="checkpoints/runs/test", files=2, size_bytes=1024)],
            size_bytes=1024,
            files=2,
            preview_token="a" * 64,
        )

    def delete(request: DeletionRequest) -> models_client.ModelJob:
        calls.append(request)
        return job()

    monkeypatch.setattr(models_client, "preview_deletion", preview)
    monkeypatch.setattr(models_client, "delete_files", delete)
    models_app.segmented_control[0].set_value("Модели").run()
    assert not calls
    next(
        button for button in models_app.button if button.label == "Показать план удаления"
    ).click().run()
    assert not models_app.exception
    assert next(
        button for button in models_app.button if button.label == "Удалить безвозвратно"
    ).disabled
    confirmation = next(
        item for item in models_app.text_input if item.label.startswith("Для подтверждения")
    )
    confirmation.set_value("УДАЛИТЬ").run()
    next(
        button for button in models_app.button if button.label == "Удалить безвозвратно"
    ).click().run()
    assert not models_app.exception
    assert len(calls) == 1 and calls[0].target.run_id == "test-run"


def test_completed_download_refreshes_catalog_automatically(
    models_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    catalog = ModelsCatalog(models=[])

    def download(model_id: str) -> models_client.ModelJob:
        calls.append(model_id)
        catalog.models.append(ManagedModel(model_id=model_id, registered=True, available=True))
        return job("completed")

    monkeypatch.setattr(models_client, "get_models", lambda: catalog)
    monkeypatch.setattr(models_client, "download_model", download)
    monkeypatch.setattr(models_client, "get_model_job", lambda _: job("completed"))
    models_app.segmented_control[0].set_value("Модели").run()
    models_app.text_input[0].set_value("org/new")
    next(button for button in models_app.button if button.label == "Загрузить модель").click().run()
    assert not models_app.exception
    assert models_app.selectbox[0].value == "org/new"
    assert calls == ["org/new"]


def test_model_controls_are_disabled_while_backend_is_busy(
    models_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = ModelsCatalog(
        models=[ManagedModel(model_id="org/model", registered=True, available=True)],
        active_job_id="training123",
    )
    monkeypatch.setattr(models_client, "get_models", lambda: catalog)
    models_app.segmented_control[0].set_value("Модели").run()
    assert not models_app.exception
    assert next(
        button for button in models_app.button if button.label == "Загрузить модель"
    ).disabled
    assert next(
        button for button in models_app.button if button.label == "Показать план удаления"
    ).disabled
