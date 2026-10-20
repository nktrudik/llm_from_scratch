"""CLI ставит задачу в существующий API, не запускает отдельную модель и не повторяет POST."""

import json
from typing import Self

import pytest
import requests

from mini_llm.inference.evaluation.cli import main
from mini_llm.ui import http


def test_cli_submits_once_and_reads_job_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    class Session:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def request(self, method: str, url: str, **kwargs: object) -> requests.Response:
            calls.append((method, url))
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps(
                {
                    "job_id": "fake",
                    "kind": "checkpoint_evaluation",
                    "status": "queued" if method == "POST" else "completed",
                }
            ).encode()
            return response

        def post(self, url: str, **kwargs: object) -> requests.Response:
            return self.request("POST", url, **kwargs)

        def get(self, url: str, **kwargs: object) -> requests.Response:
            return self.request("GET", url, **kwargs)

    monkeypatch.setattr(http, "api_session", lambda _: Session())
    assert main(["--run-id", "safe-run"]) == 0
    assert calls == [
        ("POST", "http://127.0.0.1:8000/v1/evaluations"),
        ("GET", "http://127.0.0.1:8000/v1/jobs/fake"),
    ]
