"""Polite, single-page HTML scraper for collecting raw text documents."""

from __future__ import annotations

import argparse
import hashlib
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
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup, Comment, Tag

USER_AGENT = (
    "mini-llm-data-collector/0.1 "
    "(educational single-page scraper; +https://github.com/nktrudik/llm_from_scratch)"
)
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
REMOVABLE_TAGS = (
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "form",
    "aside",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "dialog",
)
BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote")
BOILERPLATE_TOKENS = frozenset(
    {
        "ad",
        "ads",
        "advert",
        "advertisement",
        "banner",
        "breadcrumb",
        "cookie",
        "footer",
        "header",
        "menu",
        "modal",
        "nav",
        "navigation",
        "newsletter",
        "popup",
        "promo",
        "share",
        "sidebar",
        "social",
        "subscribe",
    }
)
BOILERPLATE_ROLES = frozenset({"banner", "contentinfo", "dialog", "navigation"})
TWO_CH_HOSTS = frozenset({"2ch.org", "www.2ch.org"})


class ScraperError(RuntimeError):
    """Base error raised for expected scraping failures."""


class ExtractionError(ScraperError):
    """Raised when a page does not contain enough useful text."""


@dataclass(frozen=True, slots=True)
class ScraperConfig:
    """Network, extraction, and output settings for a scraping run."""

    output_dir: Path = Path("data/raw")
    min_text_length: int = 200
    min_block_length: int = 40
    timeout_seconds: float = 15.0
    max_retries: int = 2
    backoff_factor: float = 1.0
    min_request_delay: float = 1.0
    max_request_delay: float = 2.5

    def __post_init__(self) -> None:
        if self.min_text_length <= 0:
            raise ValueError("min_text_length must be positive")
        if self.min_block_length <= 0:
            raise ValueError("min_block_length must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.backoff_factor < 0:
            raise ValueError("backoff_factor must be non-negative")
        if self.min_request_delay < 0 or self.max_request_delay < self.min_request_delay:
            raise ValueError("request delay range is invalid")


@dataclass(frozen=True, slots=True)
class ScrapedDocument:
    """Serializable text extracted from one source page."""

    source_url: str
    title: str
    text: str
    fetched_at: str

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-compatible document representation."""

        return {
            "source_url": self.source_url,
            "title": self.title,
            "text": self.text,
            "fetched_at": self.fetched_at,
        }


def normalize_text(value: str) -> str:
    """Collapse HTML whitespace into readable single spaces."""

    return re.sub(r"\s+", " ", value).strip()


def _remove_boilerplate(soup: BeautifulSoup) -> None:
    for element in soup.find_all(REMOVABLE_TAGS):
        if isinstance(element, Tag):
            element.decompose()

    for node in soup.find_all(string=True):
        if isinstance(node, Comment):
            node.extract()

    for element in list(soup.find_all(True)):
        if not isinstance(element, Tag) or element.parent is None:
            continue
        role = str(element.get("role", "")).lower()
        if role in BOILERPLATE_ROLES or element.has_attr("hidden"):
            element.decompose()
            continue
        if str(element.get("aria-hidden", "")).lower() == "true":
            element.decompose()
            continue

        identifiers = [str(element.get("id", ""))]
        classes = element.get("class", [])
        if isinstance(classes, list):
            identifiers.extend(str(class_name) for class_name in classes)
        identity_tokens = set(re.findall(r"[a-z0-9]+", " ".join(identifiers).lower()))
        if identity_tokens & BOILERPLATE_TOKENS:
            element.decompose()


def _content_score(element: Tag) -> int:
    text_length = len(normalize_text(element.get_text(" ", strip=True)))
    link_length = sum(
        len(normalize_text(link.get_text(" ", strip=True)))
        for link in element.find_all("a")
        if isinstance(link, Tag)
    )
    return max(0, text_length - 2 * link_length)


def _outermost(elements: Sequence[Tag], tag_name: str) -> list[Tag]:
    return [
        element
        for element in elements
        if not any(
            isinstance(parent, Tag) and parent.name == tag_name for parent in element.parents
        )
    ]


def _find_named_roots(soup: BeautifulSoup, tag_name: str, minimum_score: int) -> list[Tag]:
    candidates = [
        element
        for element in soup.find_all(tag_name)
        if isinstance(element, Tag) and _content_score(element) >= minimum_score
    ]
    return _outermost(candidates, tag_name)


def _find_heuristic_root(soup: BeautifulSoup, minimum_score: int) -> list[Tag]:
    candidates = [
        element
        for element in soup.find_all(("section", "div"))
        if isinstance(element, Tag) and _content_score(element) >= minimum_score
    ]
    if not candidates:
        return []
    return [max(candidates, key=_content_score)]


def _find_two_ch_post_roots(soup: BeautifulSoup) -> list[Tag]:
    """Return message bodies from the current public 2ch.org thread layout."""

    return [
        element for element in soup.select("main .post .post__message") if isinstance(element, Tag)
    ]


def _minimum_length_for(element: Tag, minimum_block_length: int) -> int:
    if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return max(4, minimum_block_length // 4)
    if element.name == "li":
        return max(20, minimum_block_length // 2)
    return minimum_block_length


def _collect_blocks(roots: Sequence[Tag], minimum_block_length: int) -> str:
    blocks: list[str] = []
    seen: set[str] = set()

    for root in roots:
        found_block = False
        for element in root.find_all(BLOCK_TAGS):
            if not isinstance(element, Tag):
                continue
            if element.name in {"li", "blockquote"} and element.find(BLOCK_TAGS):
                continue

            text = normalize_text(element.get_text(" ", strip=True))
            if len(text) < _minimum_length_for(element, minimum_block_length):
                continue
            normalized_key = text.casefold()
            if normalized_key in seen:
                continue
            seen.add(normalized_key)
            blocks.append(text)
            found_block = True

        if not found_block:
            fallback_text = normalize_text(root.get_text(" ", strip=True))
            normalized_key = fallback_text.casefold()
            if len(fallback_text) >= minimum_block_length and normalized_key not in seen:
                seen.add(normalized_key)
                blocks.append(fallback_text)

    return "\n\n".join(blocks)


def extract_document(
    html: str,
    source_url: str,
    *,
    min_text_length: int = 200,
    min_block_length: int = 40,
) -> ScrapedDocument:
    """Extract a title and deduplicated useful text from an HTML document."""

    if min_text_length <= 0 or min_block_length <= 0:
        raise ValueError("minimum text lengths must be positive")

    soup = BeautifulSoup(html, "html.parser")
    title_element = soup.find("title")
    title = (
        normalize_text(title_element.get_text(" ", strip=True))
        if isinstance(title_element, Tag)
        else ""
    )
    _remove_boilerplate(soup)
    if not title:
        heading = soup.find("h1")
        if isinstance(heading, Tag):
            title = normalize_text(heading.get_text(" ", strip=True))

    root_groups: list[tuple[list[Tag], int]] = []
    if (urlparse(source_url).hostname or "").lower() in TWO_CH_HOSTS:
        two_ch_block_length = max(10, min_block_length // 2)
        root_groups.append((_find_two_ch_post_roots(soup), two_ch_block_length))

    root_groups.extend(
        [
            (_find_named_roots(soup, "article", min_block_length), min_block_length),
            (_find_named_roots(soup, "main", min_block_length), min_block_length),
            (_find_heuristic_root(soup, min_block_length), min_block_length),
        ]
    )
    if isinstance(soup.body, Tag):
        root_groups.append(([soup.body], min_block_length))

    longest_text = ""
    for roots, block_length in root_groups:
        if not roots:
            continue
        text = _collect_blocks(roots, block_length)
        if len(text) > len(longest_text):
            longest_text = text
        if len(text) >= min_text_length:
            return ScrapedDocument(
                source_url=source_url,
                title=title,
                text=text,
                fetched_at=datetime.now(UTC).isoformat(),
            )

    raise ExtractionError(
        f"Extracted text is too short: {len(longest_text)} characters (minimum: {min_text_length})"
    )


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


def _validate_url(url: str) -> None:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ScraperError("URL must be an absolute http:// or https:// address")


def _output_filename(url: str) -> str:
    parsed_url = urlparse(url)
    readable_part = unquote(f"{parsed_url.hostname or 'page'}{parsed_url.path}")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", readable_part).strip("-").lower()
    slug = slug[:80] or "page"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}.json"


def save_document(document: ScrapedDocument, output_dir: Path) -> Path:
    """Save a scraped document as UTF-8 JSON and return its path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _output_filename(document.source_url)
    output_path.write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


class WebScraper:
    """Sequential HTML scraper backed by one reusable requests session."""

    def __init__(self, config: ScraperConfig | None = None) -> None:
        self.config = config or ScraperConfig()
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
        )
        self._last_request_finished_at: float | None = None

    def __enter__(self) -> WebScraper:
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

    def fetch_html(self, url: str) -> str:
        """Download one HTML page with bounded retries and polite delays."""

        _validate_url(url)
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
            if content_type and "html" not in content_type and "xhtml" not in content_type:
                raise ScraperError(f"URL did not return HTML content: {content_type}")
            response.encoding = response.apparent_encoding or response.encoding
            return response.text

        raise ScraperError(f"Request failed after bounded retries: {url}") from last_error

    def scrape(self, url: str) -> ScrapedDocument:
        """Download and extract one document without saving it."""

        html = self.fetch_html(url)
        return extract_document(
            html,
            url,
            min_text_length=self.config.min_text_length,
            min_block_length=self.config.min_block_length,
        )

    def scrape_to_file(self, url: str) -> Path:
        """Download, extract, and save one document to the configured raw-data directory."""

        return save_document(self.scrape(url), self.config.output_dir)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Extract useful text from one ordinary HTML page.")
    parser.add_argument("url", help="Absolute http:// or https:// page URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for the resulting JSON file (default: data/raw)",
    )
    parser.add_argument("--min-text-length", type=int, default=200)
    parser.add_argument("--min-block-length", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=15.0, help="Request timeout in seconds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the single-page scraper CLI."""

    args = build_argument_parser().parse_args(argv)
    try:
        config = ScraperConfig(
            output_dir=args.output_dir,
            min_text_length=args.min_text_length,
            min_block_length=args.min_block_length,
            timeout_seconds=args.timeout,
        )
        with WebScraper(config) as scraper:
            output_path = scraper.scrape_to_file(args.url)
    except (ScraperError, ValueError) as error:
        print(f"Scraping failed: {error}")
        return 1

    print(f"Saved extracted document to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
