"""Последовательный HTTP-клиент публичного JSON API 2ch."""

from __future__ import annotations

import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import TracebackType
from typing import Self, cast

import requests

from mini_llm.data.scraping.parsing import (
    parse_catalog_thread_ids,
    parse_thread_document,
    parse_two_ch_url,
)
from mini_llm.data.scraping.schemas import (
    RETRYABLE_STATUS_CODES,
    TWO_CH_BASE_URL,
    USER_AGENT,
    ScraperConfig,
    ScraperError,
    ThreadDocument,
)
from mini_llm.data.scraping.storage import save_thread_document, thread_output_path


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


class TwoChScraper:
    """Последовательный клиент 2ch JSON API с одной HTTP-сессией."""

    def __init__(self, config: ScraperConfig | None = None) -> None:
        self.config = config or ScraperConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._last_request_finished_at: float | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Закрыть используемую HTTP-сессию."""

        self.session.close()

    def _wait_before_request(self, minimum_wait: float) -> None:
        if self._last_request_finished_at is None:
            if minimum_wait > 0:
                time.sleep(minimum_wait)
            return

        random_interval = random.uniform(  # noqa: S311  # nosec B311
            self.config.min_request_delay,
            self.config.max_request_delay,
        )
        elapsed = time.monotonic() - self._last_request_finished_at
        wait_seconds = max(minimum_wait, random_interval - elapsed)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def fetch_json(self, url: str) -> object:
        """Получить JSON с ограниченными retries, backoff и вежливой задержкой."""

        next_wait = 0.0
        last_error: requests.RequestException | None = None

        for attempt in range(self.config.max_retries + 1):
            self._wait_before_request(next_wait)
            try:
                response = self.session.get(url, timeout=self.config.timeout_seconds)
            except requests.RequestException as error:
                last_error = error
                self._last_request_finished_at = time.monotonic()
                if attempt >= self.config.max_retries:
                    break
                next_wait = self.config.backoff_factor * (2**attempt)
                continue

            self._last_request_finished_at = time.monotonic()
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.config.max_retries:
                retry_after = (
                    _retry_after_seconds(response.headers.get("Retry-After"))
                    if response.status_code == 429
                    else None
                )
                next_wait = (
                    retry_after
                    if retry_after is not None
                    else self.config.backoff_factor * (2**attempt)
                )
                continue

            try:
                response.raise_for_status()
            except requests.RequestException as error:
                raise ScraperError(f"HTTP-запрос завершился ошибкой для {url}: {error}") from error

            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and "json" not in content_type:
                raise ScraperError(f"API endpoint вернул не JSON: {content_type}")
            try:
                return cast(object, response.json())
            except ValueError as error:
                raise ScraperError(f"API endpoint вернул некорректный JSON: {url}") from error

        raise ScraperError(f"Запрос не выполнен после ограниченных retries: {url}") from last_error

    def scrape_thread(self, board: str, thread_id: int) -> ThreadDocument:
        """Скачать и разобрать один полный тред."""

        api_url = f"{TWO_CH_BASE_URL}/{board}/res/{thread_id}.json"
        return parse_thread_document(self.fetch_json(api_url), board, thread_id)

    def scrape_to_files(self, url: str, *, max_threads: int | None = None) -> list[Path]:
        """Собрать один тред или выбранные актуальные треды доски."""

        target = parse_two_ch_url(url)
        if max_threads is not None and max_threads <= 0:
            raise ValueError("max_threads должен быть положительным")

        if target.thread_id is not None:
            candidate_ids = [target.thread_id]
        else:
            catalog_url = f"{TWO_CH_BASE_URL}/{target.board}/catalog.json"
            catalog = self.fetch_json(catalog_url)
            candidate_ids = parse_catalog_thread_ids(catalog)

        output_paths: list[Path] = []
        for thread_id in candidate_ids:
            if thread_output_path(self.config.output_dir, target.board, thread_id).exists():
                continue
            document = self.scrape_thread(target.board, thread_id)
            output_path = save_thread_document(document, self.config.output_dir)
            if output_path is not None:
                output_paths.append(output_path)
            if max_threads is not None and len(output_paths) >= max_threads:
                break
        return output_paths
