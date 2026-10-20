"""Терминальный клиент той же очереди API, без второго GPU-процесса."""

import argparse
import time
from collections.abc import Sequence

import requests

from mini_llm.inference.evaluation.schemas import EvaluationJob, EvaluationRequest


def main(argv: Sequence[str] | None = None) -> int:
    """Поставить прогон в API и ждать статуса; Ctrl+C запрашивает безопасную остановку."""

    parser = argparse.ArgumentParser(description="Проверить все checkpoints через запущенный API")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    request = EvaluationRequest(run_id=args.run_id, device=args.device)
    base = args.api_url.rstrip("/")
    job: EvaluationJob | None = None
    # Одна Session; локальные запросы не должны уходить в системный proxy.
    from mini_llm.ui.http import api_session, response_payload

    try:
        with api_session(base) as session:
            response = session.post(
                base + "/v1/evaluations", json=request.model_dump(), timeout=(5, 15)
            )
            job = EvaluationJob.model_validate(response_payload(response))
            print(f"Запущено: {job.job_id}")
            last_message = ""
            while job.status in {"queued", "running"}:
                response = session.get(base + f"/v1/jobs/{job.job_id}", timeout=(5, 15))
                job = EvaluationJob.model_validate(response_payload(response))
                if job.evaluation_progress:
                    snapshot = job.evaluation_progress
                    message = (
                        f"{snapshot.completed}/{snapshot.total} · "
                        f"{snapshot.current_checkpoint} · {snapshot.stage}"
                    )
                    if message != last_message:
                        print(message)
                        last_message = message
                if job.status in {"queued", "running"}:
                    time.sleep(1)
            if job.evaluation_progress:
                print(f"Отчёт: {job.evaluation_progress.report_file}")
            if job.status == "failed":
                print(job.error)
                return 1
    except KeyboardInterrupt:
        if job is not None:
            with api_session(base) as session:
                session.post(
                    base + f"/v1/jobs/{job.job_id}/stop", timeout=(5, 15)
                ).raise_for_status()
            print(f"Остановка запрошена; текущий ответ будет сохранён. Задача: {job.job_id}")
        return 130
    except (requests.RequestException, ValueError, RuntimeError) as error:
        print(f"Ошибка API: {error}. Проверьте backend; повторный запуск не выполнялся.")
        return 1
    return 0
