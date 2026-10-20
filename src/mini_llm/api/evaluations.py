"""Ручной запуск проверки checkpoints через общий GPU-слот фоновых задач."""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from mini_llm.api.jobs import JobAlreadyRunningError, JobManager
from mini_llm.api.schemas import JobResponse
from mini_llm.inference.evaluation.pipeline import evaluation_plan, run_evaluation
from mini_llm.inference.evaluation.report import REPORTS_ROOT
from mini_llm.inference.evaluation.schemas import QUESTIONS, EvaluationProgress, EvaluationRequest
from mini_llm.workspace.catalog import contained


def create_evaluation_router(manager: JobManager) -> APIRouter:
    """Не запускать генерацию при открытии страницы или чтении списка checkpoints."""

    router = APIRouter(prefix="/v1/evaluations", tags=["Проверка checkpoints"])

    @router.get("/plan")
    def plan(run_id: str) -> dict[str, object]:
        """Перечислить реальные файлы, включая best.pt и last.pt, без загрузки весов."""

        try:
            run, paths = evaluation_plan(EvaluationRequest(run_id=run_id))
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "run_id": run.run_id,
            "checkpoints": [p.name for p in paths],
            "questions": list(QUESTIONS),
            "total": len(paths) * len(QUESTIONS),
        }

    @router.post("", response_model=JobResponse, status_code=202)
    def start(request: EvaluationRequest, tasks: BackgroundTasks) -> JobResponse:
        """Занять общий слот до начала работы; чат, training и удаление будут заблокированы."""

        try:
            evaluation_plan(request)
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        try:
            record = manager.create("checkpoint_evaluation")
        except JobAlreadyRunningError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

        def progress(snapshot: EvaluationProgress) -> None:
            manager.report_evaluation(record.job_id, snapshot.model_dump())

        tasks.add_task(
            manager.run, record.job_id, lambda: run_evaluation(request, report_progress=progress)
        )
        return JobResponse.model_validate(record.to_dict())

    @router.get("/{job_id}/report", response_class=FileResponse)
    def report(job_id: str) -> FileResponse:
        """Отдать сохранённый Markdown только по серверному пути известной задачи."""

        from pathlib import Path

        job = manager.get(job_id)
        snapshot = job.get("evaluation_progress") if job else None
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("report_file"), str):
            raise HTTPException(status_code=404, detail="Отчёт ещё не создан")
        try:
            path = contained(Path(snapshot["report_file"]), REPORTS_ROOT)
            if not path.is_file():
                raise ValueError("Файл отчёта не найден")
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if job and job.get("status") in {"queued", "running"}:
            raise HTTPException(
                status_code=409,
                detail="Отчёт дописывается; дождитесь завершения или остановите прогон",
            )
        return FileResponse(path, media_type="text/markdown", filename=path.name)

    return router
