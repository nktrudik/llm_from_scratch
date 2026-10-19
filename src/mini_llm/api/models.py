"""Локальные ручки каталога, загрузки и безопасной очистки моделей."""

from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import APIRouter, BackgroundTasks, HTTPException

from mini_llm.api.jobs import JobAlreadyRunningError, JobManager
from mini_llm.api.schemas import JobResponse
from mini_llm.inference import clear_pretrained_cache
from mini_llm.pretrained.management.catalog import models_catalog
from mini_llm.pretrained.management.schemas import (
    DeletionPreview,
    DeletionRequest,
    DeletionTarget,
    DownloadRequest,
    ModelsCatalog,
    OperationProgress,
)
from mini_llm.pretrained.management.service import delete_model_files, download_model
from mini_llm.pretrained.management.storage import deletion_preview


def create_model_router(manager: JobManager) -> APIRouter:
    """Использовать общую очередь с training и блокировку генерации."""

    router = APIRouter(prefix="/v1/models", tags=["Модели"])

    def submit(
        kind: str,
        tasks: BackgroundTasks,
        operation: Callable[[Callable[[str], None]], dict[str, object]],
    ) -> JobResponse:
        try:
            record = manager.create(kind)
        except JobAlreadyRunningError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

        def execute() -> dict[str, object]:
            started = time.perf_counter()

            def report(stage: str) -> None:
                manager.report_operation(
                    record.job_id,
                    OperationProgress(
                        stage=stage,
                        elapsed_seconds=time.perf_counter() - started,
                    ).model_dump(),
                )

            report("Освобождение кэша модели в памяти")
            clear_pretrained_cache()
            return operation(report)

        tasks.add_task(manager.run, record.job_id, execute)
        return JobResponse.model_validate(record.to_dict())

    @router.get("", response_model=ModelsCatalog)
    def catalog() -> ModelsCatalog:
        """Обновить локальный список и размер; не обращаться к сети и не читать веса."""

        try:
            return models_catalog(manager.active_job_id())
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.post("/download", response_model=JobResponse, status_code=202)
    def download(request: DownloadRequest, tasks: BackgroundTasks) -> JobResponse:
        """Начать скачивание только после явного действия пользователя."""

        return submit("model_download", tasks, lambda report: download_model(request, report))

    @router.post("/deletion-preview", response_model=DeletionPreview)
    def preview(target: DeletionTarget) -> DeletionPreview:
        """Показать точные пути и объём до подтверждения удаления."""

        try:
            return deletion_preview(target)
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.post("/delete", response_model=JobResponse, status_code=202)
    def delete(request: DeletionRequest, tasks: BackgroundTasks) -> JobResponse:
        """Заблокировать остальные операции до проверки и завершения очистки."""

        return submit("model_delete", tasks, lambda report: delete_model_files(request, report))

    return router
