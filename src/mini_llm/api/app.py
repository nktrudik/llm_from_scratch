"""FastAPI-приложение для ручного запуска pipeline и генерации."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import BackgroundTasks, FastAPI, HTTPException, status

from mini_llm.api.jobs import JobAlreadyRunningError, JobManager
from mini_llm.api.schemas import (
    DatasetSplitRequest,
    GenerationOptionsResponse,
    GenerationRequest,
    GenerationResponse,
    HealthResponse,
    JobResponse,
    PreprocessingRequest,
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
from mini_llm.inference.options import pretrained_generation_options

app = FastAPI(
    title="mini_llm local control API",
    description="Локальный API для последовательного запуска этапов проекта.",
    version="0.1.0",
)
job_manager = JobManager()


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
    return JobResponse.model_validate(snapshot)


@app.post("/v1/training", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_training(request: TrainingRequest, background_tasks: BackgroundTasks) -> JobResponse:
    """Поставить полное обучение модели в локальную фоновую очередь."""

    return _submit_job("training", background_tasks, lambda: run_training(request))


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
    """Проверить доступность до/после SFT, не загружая веса в CPU или GPU."""

    availability = pretrained_generation_options()
    return GenerationOptionsResponse(
        model_id=availability.model_id,
        before_sft_available=availability.before_sft_available,
        after_sft_available=availability.after_sft_available,
        reason=availability.reason,
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
        result = generate_response(request.prompt, request.to_config())
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    return GenerationResponse(text=result.text, token_ids=result.token_ids)
