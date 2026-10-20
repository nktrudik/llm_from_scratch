"""FastAPI-приложение для ручного запуска pipeline и генерации."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from fastapi import BackgroundTasks, FastAPI, HTTPException, status

from mini_llm.api.evaluations import create_evaluation_router
from mini_llm.api.jobs import JobAlreadyRunningError, JobManager
from mini_llm.api.models import create_model_router
from mini_llm.api.schemas import (
    DatasetSplitRequest,
    ExperimentOptionsResponse,
    GenerationOptionsResponse,
    GenerationRequest,
    GenerationResponse,
    HealthResponse,
    JobResponse,
    PreprocessingRequest,
    PretrainedModelOptionsResponse,
    PretrainedPrepareRequest,
    ScrapeRequest,
    TokenizerTrainingRequest,
    TokenStatisticsRequest,
    TrainingRequest,
)
from mini_llm.api.services import (
    run_dataset_split,
    run_preprocessing,
    run_pretrained_prepare,
    run_scraper,
    run_token_statistics,
    run_tokenizer_training,
    run_training,
)
from mini_llm.inference import clear_pretrained_cache, generate_response
from mini_llm.inference.experiments import list_experiments
from mini_llm.inference.options import pretrained_generation_options
from mini_llm.modeling.config import ModelConfig
from mini_llm.observability.training import read_training_progress
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.workspace import list_registered_models
from mini_llm.workspace.catalog import list_runs
from mini_llm.workspace.discovery import dataset_catalog
from mini_llm.workspace.training import (
    ManagedTrainingRequest,
    prepare_version_splits,
    run_managed_training,
    validate_training_request,
)

app = FastAPI(
    title="mini_llm local control API",
    description="Локальный API для последовательного запуска этапов проекта.",
    version="0.1.0",
)
job_manager = JobManager()
app.include_router(create_model_router(job_manager))
app.include_router(create_evaluation_router(job_manager))


@app.get("/v1/workspace")
def workspace_catalog(refresh_datasets: bool = True) -> dict[str, object]:
    """Обновить версии по запросу; считать samples только у новых папок, без загрузки весов."""

    datasets, warnings = dataset_catalog(refresh=refresh_datasets)
    models: list[dict[str, object]] = [
        {
            "model_id": "custom",
            "max_sequence_length": ModelConfig().max_sequence_length,
            "adaptation_mode": "full",
        }
    ]
    for registration in list_registered_models():
        config = PretrainedConfig.load(registration.config_file)
        models.append(
            {
                "model_id": registration.model_id,
                "max_sequence_length": config.max_sequence_length,
                "adaptation_mode": config.adaptation_mode,
            }
        )
    return {
        "datasets": [{**item.model_dump(), "splits_ready": item.splits_ready} for item in datasets],
        "dataset_count": len(datasets),
        "total_samples": sum(item.samples for item in datasets),
        "dataset_warnings": warnings,
        "models": models,
        "runs": [item.model_dump(mode="json") for item in list_runs()],
        "active_job_id": job_manager.active_job_id(),
    }


@app.post("/v1/workspace/training", response_model=JobResponse, status_code=202)
def start_versioned_training(
    request: ManagedTrainingRequest, background_tasks: BackgroundTasks
) -> JobResponse:
    """Запустить обучение по версиям, сохранив существующие запуски."""

    try:
        validate_training_request(request)
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        record = job_manager.create("training")
    except JobAlreadyRunningError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    clear_pretrained_cache()
    background_tasks.add_task(
        job_manager.run, record.job_id, lambda: run_managed_training(request, record.job_id)
    )
    return JobResponse.model_validate(record.to_dict())


@app.post(
    "/v1/workspace/datasets/{dataset_id}/{version}/split",
    response_model=JobResponse,
    status_code=202,
)
def start_versioned_split(
    dataset_id: str, version: str, background_tasks: BackgroundTasks
) -> JobResponse:
    """Явно подготовить splits версии, не изменяя другие версии."""

    return _submit_job(
        "dataset_split", background_tasks, lambda: prepare_version_splits(dataset_id, version)
    )


def _submit_job(
    kind: str,
    background_tasks: BackgroundTasks,
    operation: Callable[[], object],
) -> JobResponse:
    try:
        record = job_manager.create(kind)
    except JobAlreadyRunningError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    if kind in {"training", "pretrained_prepare"}:
        clear_pretrained_cache()
    background_tasks.add_task(job_manager.run, record.job_id, operation)
    return JobResponse.model_validate(record.to_dict())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Проверить доступность API и наличие активной фоновой задачи."""

    return HealthResponse(status="ok", active_job_id=job_manager.active_job_id())


@app.get("/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    """Получить состояние фоновой задачи по идентификатору."""

    snapshot = job_manager.get(job_id)
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Задача не найдена")
    if snapshot.get("kind") == "training":
        run = next((item for item in list_runs() if item.job_id == job_id), None)
        if run:
            snapshot["progress"] = read_training_progress(run.directory)
    return JobResponse.model_validate(snapshot)


@app.post("/v1/training", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_training(request: TrainingRequest, background_tasks: BackgroundTasks) -> JobResponse:
    """Поставить полное обучение модели в локальную фоновую очередь."""

    return _submit_job("training", background_tasks, lambda: run_training(request))


@app.post("/v1/jobs/{job_id}/stop", response_model=JobResponse)
def stop_training(job_id: str) -> JobResponse:
    """Запросить сохранение и остановку на ближайшей безопасной границе шага."""

    try:
        return JobResponse.model_validate(job_manager.request_stop(job_id))
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Задача не найдена") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post(
    "/v1/pretrained/prepare",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_pretrained_prepare(
    request: PretrainedPrepareRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Загрузить pretrained model в cache и подготовить режим адаптации."""

    return _submit_job(
        "pretrained_prepare",
        background_tasks,
        lambda: run_pretrained_prepare(request),
    )


@app.post(
    "/v1/tokenizer/train",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_tokenizer_training(
    request: TokenizerTrainingRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Поставить обучение BPE tokenizer в локальную фоновую очередь."""

    return _submit_job(
        "tokenizer_training",
        background_tasks,
        lambda: run_tokenizer_training(request),
    )


@app.post("/v1/preprocessing", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_preprocessing(
    request: PreprocessingRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Поставить preprocessing raw-корпуса в локальную фоновую очередь."""

    return _submit_job("preprocessing", background_tasks, lambda: run_preprocessing(request))


@app.post("/v1/dataset/split", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_dataset_split(
    request: DatasetSplitRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Поставить thread-level split в локальную фоновую очередь."""

    return _submit_job("dataset_split", background_tasks, lambda: run_dataset_split(request))


@app.post(
    "/v1/statistics/tokens",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_token_statistics(
    request: TokenStatisticsRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Поставить подсчёт token statistics в локальную фоновую очередь."""

    return _submit_job(
        "token_statistics",
        background_tasks,
        lambda: run_token_statistics(request),
    )


@app.post("/v1/scraper", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_scraper(request: ScrapeRequest, background_tasks: BackgroundTasks) -> JobResponse:
    """Поставить последовательный сбор тредов в локальную фоновую очередь."""

    return _submit_job("scraper", background_tasks, lambda: run_scraper(request))


@app.get("/v1/generate/options", response_model=GenerationOptionsResponse)
def generation_options() -> GenerationOptionsResponse:
    """Перечислить модели и режимы, не загружая веса в CPU или GPU."""

    availability = pretrained_generation_options()
    try:
        registrations = list_registered_models()
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    models = []
    for registration in registrations:
        options = pretrained_generation_options(registration.model_id)
        models.append(
            PretrainedModelOptionsResponse(
                model_id=options.model_id,
                before_sft_available=options.before_sft_available,
                after_sft_available=options.after_sft_available,
                reason=options.reason,
                experiments=[
                    ExperimentOptionsResponse.model_validate(asdict(item))
                    for item in options.experiments
                ],
            )
        )
    return GenerationOptionsResponse(
        model_id=availability.model_id,
        before_sft_available=availability.before_sft_available,
        after_sft_available=availability.after_sft_available,
        reason=availability.reason,
        experiments=[
            ExperimentOptionsResponse.model_validate(asdict(item))
            for item in availability.experiments
        ],
        models=models,
        custom_runs=[
            ExperimentOptionsResponse.model_validate(asdict(item))
            for item in list_experiments(None)
        ],
    )


@app.post("/v1/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest) -> GenerationResponse:
    """Сгенерировать один ответ, повторно используя pretrained-модель в памяти."""

    active_job_id = job_manager.active_job_id()
    if active_job_id is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Генерация недоступна, пока выполняется задача {active_job_id}",
        )
    try:
        with job_manager.generation():
            result = generate_response(request.prompt, request.to_config())
    except JobAlreadyRunningError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    return GenerationResponse(text=result.text, token_ids=result.token_ids)
