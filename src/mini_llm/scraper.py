"""Polite, sequential collector for public 2ch JSON API threads."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import TracebackType
from typing import Self, cast
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag

TWO_CH_BASE_URL = "https://2ch.org"
TWO_CH_HOSTS = frozenset({"2ch.org", "www.2ch.org"})
USER_AGENT = (
    "mini-llm-data-collector/0.1 "
    "(educational sequential collector; +https://github.com/nktrudik/llm_from_scratch)"
)
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
THREAD_PATH_PATTERN = re.compile(r"^/(?P<board>[a-zA-Z0-9]+)/res/(?P<thread_id>\d+)\.html/?$")
BOARD_PATH_PATTERN = re.compile(r"^/(?P<board>[a-zA-Z0-9]+)/?$")
REFERENCE_PATTERN = re.compile(r">>(\d+)")


class ScraperError(RuntimeError):
    """Error raised for an expected request, URL, or API schema failure."""


@dataclass(frozen=True, slots=True)
class ScraperConfig:
    """Network and output settings for a collection run."""

    output_dir: Path = Path("data/raw")
    timeout_seconds: float = 15.0
    max_retries: int = 2
    backoff_factor: float = 1.0
    min_request_delay: float = 1.0
    max_request_delay: float = 2.5

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.backoff_factor < 0:
            raise ValueError("backoff_factor must be non-negative")
        if self.min_request_delay < 0 or self.max_request_delay < self.min_request_delay:
            raise ValueError("request delay range is invalid")


@dataclass(frozen=True, slots=True)
class TwoChTarget:
    """Parsed board URL with an optional concrete thread ID."""

    board: str
    thread_id: int | None = None


@dataclass(frozen=True, slots=True)
class PostRecord:
    """Text-only representation of one non-empty 2ch post."""

    post_id: int
    text: str
    references: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible post representation."""

        return {
            "post_id": self.post_id,
            "text": self.text,
            "references": list(self.references),
        }


@dataclass(frozen=True, slots=True)
class ThreadDocument:
    """Text-only thread collected from the 2ch JSON API."""

    board: str
    thread_id: int
    source_url: str
    api_url: str
    title: str
    fetched_at: str
    posts: tuple[PostRecord, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible thread representation."""

        return {
            "board": self.board,
            "thread_id": self.thread_id,
            "source_url": self.source_url,
            "api_url": self.api_url,
            "title": self.title,
            "fetched_at": self.fetched_at,
            "posts": [post.to_dict() for post in self.posts],
        }


def parse_two_ch_url(url: str) -> TwoChTarget:
    """Parse a supported 2ch board or thread URL."""

    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or parsed_url.hostname not in TWO_CH_HOSTS:
        raise ScraperError("URL must point to a public board or thread on 2ch.org")

    thread_match = THREAD_PATH_PATTERN.fullmatch(parsed_url.path)
    if thread_match:
        return TwoChTarget(
            board=thread_match.group("board").lower(),
            thread_id=int(thread_match.group("thread_id")),
        )

    board_match = BOARD_PATH_PATTERN.fullmatch(parsed_url.path)
    if board_match:
        return TwoChTarget(board=board_match.group("board").lower())

    raise ScraperError(
        "Expected a board URL like https://2ch.org/b/ or a thread URL like "
        "https://2ch.org/b/res/123.html"
    )


def clean_post_text(html: str) -> str:
    """Convert the HTML fragment in an API comment into readable plain text."""

    soup = BeautifulSoup(html, "html.parser")
    for element in soup.find_all(("script", "style")):
        if isinstance(element, Tag):
            element.decompose()
    for line_break in soup.find_all("br"):
        if isinstance(line_break, Tag):
            line_break.replace_with("\n")

    raw_text = soup.get_text(" ", strip=False)
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw_text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_references(text: str) -> tuple[str, ...]:
    """Return unique post references in their original ``>>123`` form."""

    return tuple(dict.fromkeys(f">>{post_id}" for post_id in REFERENCE_PATTERN.findall(text)))


def _require_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ScraperError(f"Invalid 2ch API response: {context} must be an object")
    return cast(dict[str, object], value)


def _require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ScraperError(f"Invalid 2ch API response: {context} must be a list")
    return cast(list[object], value)


def _require_integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ScraperError(f"Invalid 2ch API response: {context} must be an integer")
    return value


def parse_catalog_thread_ids(payload: object, max_threads: int | None = None) -> list[int]:
    """Extract ordered thread IDs from a board catalog API response."""

    if max_threads is not None and max_threads <= 0:
        raise ValueError("max_threads must be positive when provided")

    root = _require_object(payload, "catalog root")
    raw_threads = _require_list(root.get("threads"), "threads")
    thread_ids: list[int] = []
    seen: set[int] = set()
    for index, raw_thread in enumerate(raw_threads):
        thread = _require_object(raw_thread, f"threads[{index}]")
        thread_id = _require_integer(thread.get("num"), f"threads[{index}].num")
        if thread_id in seen:
            continue
        seen.add(thread_id)
        thread_ids.append(thread_id)
        if max_threads is not None and len(thread_ids) >= max_threads:
            break

    return thread_ids


def parse_thread_document(payload: object, board: str, thread_id: int) -> ThreadDocument:
    """Create a text-only thread document from a 2ch API response."""

    root = _require_object(payload, "thread root")
    raw_threads = _require_list(root.get("threads"), "threads")
    if not raw_threads:
        raise ScraperError("Invalid 2ch API response: thread list is empty")
    thread = _require_object(raw_threads[0], "threads[0]")
    raw_posts = _require_list(thread.get("posts"), "threads[0].posts")

    posts: list[PostRecord] = []
    title = str(root.get("title", "")).strip()
    for index, raw_post in enumerate(raw_posts):
        post = _require_object(raw_post, f"posts[{index}]")
        raw_comment = post.get("comment")
        if not isinstance(raw_comment, str):
            continue
        text = clean_post_text(raw_comment)
        if not text:
            continue

        post_id = _require_integer(post.get("num"), f"posts[{index}].num")
        posts.append(
            PostRecord(
                post_id=post_id,
                text=text,
                references=extract_references(text),
            )
        )
        subject = post.get("subject")
        if len(posts) == 1 and isinstance(subject, str) and subject.strip():
            title = clean_post_text(subject)

    source_url = f"{TWO_CH_BASE_URL}/{board}/res/{thread_id}.html"
    api_url = f"{TWO_CH_BASE_URL}/{board}/res/{thread_id}.json"
    return ThreadDocument(
        board=board,
        thread_id=thread_id,
        source_url=source_url,
        api_url=api_url,
        title=title,
        fetched_at=datetime.now(UTC).isoformat(),
        posts=tuple(posts),
    )


def save_thread_document(document: ThreadDocument, output_dir: Path) -> Path:
    """Save one thread under ``data/raw/2ch/<board>/<thread_id>.json``."""

    board_dir = output_dir / "2ch" / document.board
    board_dir.mkdir(parents=True, exist_ok=True)
    output_path = board_dir / f"{document.thread_id}.json"
    output_path.write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


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
    """Sequential 2ch JSON API client backed by one reusable session."""

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
        """Close the underlying HTTP session."""

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
        """Fetch one JSON endpoint with bounded retries and polite delays."""

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
                raise ScraperError(f"HTTP request failed for {url}: {error}") from error

            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and "json" not in content_type:
                raise ScraperError(f"API endpoint did not return JSON: {content_type}")
            try:
                return cast(object, response.json())
            except ValueError as error:
                raise ScraperError(f"API endpoint returned invalid JSON: {url}") from error

        raise ScraperError(f"Request failed after bounded retries: {url}") from last_error

    def scrape_thread(self, board: str, thread_id: int) -> ThreadDocument:
        """Download and parse one complete thread."""

        api_url = f"{TWO_CH_BASE_URL}/{board}/res/{thread_id}.json"
        return parse_thread_document(self.fetch_json(api_url), board, thread_id)

    def scrape_to_files(self, url: str, *, max_threads: int | None = None) -> list[Path]:
        """Collect one thread or all selected current threads from a board URL."""

        target = parse_two_ch_url(url)
        if max_threads is not None and max_threads <= 0:
            raise ValueError("max_threads must be positive when provided")

        if target.thread_id is not None:
            thread_ids = [target.thread_id]
        else:
            catalog_url = f"{TWO_CH_BASE_URL}/{target.board}/catalog.json"
            catalog = self.fetch_json(catalog_url)
            thread_ids = parse_catalog_thread_ids(catalog, max_threads)

        output_paths: list[Path] = []
        for thread_id in thread_ids:
            document = self.scrape_thread(target.board, thread_id)
            output_paths.append(save_thread_document(document, self.config.output_dir))
        return output_paths


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Collect text-only threads from the 2ch JSON API.")
    parser.add_argument("url", help="2ch board URL or concrete thread URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw-data root directory (default: data/raw)",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Maximum threads to collect from a board URL",
    )
    parser.add_argument("--timeout", type=float, default=15.0, help="Request timeout in seconds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the 2ch JSON collector CLI."""

    args = build_argument_parser().parse_args(argv)
    try:
        config = ScraperConfig(output_dir=args.output_dir, timeout_seconds=args.timeout)
        with TwoChScraper(config) as scraper:
            output_paths = scraper.scrape_to_files(args.url, max_threads=args.max_threads)
    except (ScraperError, ValueError) as error:
        print(f"Scraping failed: {error}")
        return 1

    print(f"Saved {len(output_paths)} thread(s) under {config.output_dir / '2ch'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
