"""Короткие HTTP-запросы проверки checkpoints; inference выполняется только в API."""

from urllib.parse import quote

import requests
from pydantic import BaseModel, ValidationError

from mini_llm.inference.evaluation.schemas import EvaluationJob
from mini_llm.ui.config import API_BASE_URL, OPTIONS_TIMEOUT
from mini_llm.ui.http import APIError, api_session, response_payload


class EvaluationPlan(BaseModel):
    """Количество checkpoint-файлов и вопросов до запуска GPU-задачи."""

    run_id: str
    checkpoints: list[str]
    questions: list[str]
    total: int


def _request[T: BaseModel](
    schema: type[T], method: str, path: str, body: dict[str, object] | None = None
) -> T:
    try:
        with api_session(API_BASE_URL) as session:
            response = session.request(
                method, API_BASE_URL + path, json=body, timeout=OPTIONS_TIMEOUT
            )
        return schema.model_validate(response_payload(response))
    except requests.RequestException as error:
        raise APIError(f"API недоступен ({API_BASE_URL}): {error}") from error
    except ValidationError as error:
        raise APIError(
            "API вернул некорректный прогресс проверки; перезапустите backend"
        ) from error


def get_plan(run_id: str) -> EvaluationPlan:
    """План не запускает генерацию и не обращается к Hugging Face."""

    return _request(EvaluationPlan, "GET", f"/v1/evaluations/plan?run_id={quote(run_id, safe='')}")


def start_evaluation(run_id: str) -> EvaluationJob:
    """Отправить запуск ровно один раз, без автоматических повторов POST."""

    return _request(EvaluationJob, "POST", "/v1/evaluations", {"run_id": run_id})


def get_job(job_id: str) -> EvaluationJob:
    """Прочитать лёгкий статус вместо ожидания всего прогона в HTTP-запросе."""

    return _request(EvaluationJob, "GET", f"/v1/jobs/{quote(job_id, safe='')}")


def stop_evaluation(job_id: str) -> EvaluationJob:
    """Завершить текущий ответ, сохранить его и остановиться перед следующим."""

    return _request(EvaluationJob, "POST", f"/v1/jobs/{quote(job_id, safe='')}/stop")


def report_url(job_id: str) -> str:
    """Скачать отчёт через API, не читая серверные файлы на стороне UI."""

    return f"{API_BASE_URL}/v1/evaluations/{quote(job_id, safe='')}/report"
