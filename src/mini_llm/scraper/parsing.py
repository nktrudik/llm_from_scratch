"""Разбор URL и ответов JSON API 2ch."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import cast
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from mini_llm.scraper.schemas import (
    TWO_CH_BASE_URL,
    TWO_CH_HOSTS,
    PostRecord,
    ScraperError,
    ThreadDocument,
    TwoChTarget,
)

THREAD_PATH_PATTERN = re.compile(r"^/(?P<board>[a-zA-Z0-9]+)/res/(?P<thread_id>\d+)\.html/?$")
BOARD_PATH_PATTERN = re.compile(r"^/(?P<board>[a-zA-Z0-9]+)/?$")
REFERENCE_PATTERN = re.compile(r">>(\d+)")


def parse_two_ch_url(url: str) -> TwoChTarget:
    """Разобрать поддерживаемый URL доски или треда 2ch."""

    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or parsed_url.hostname not in TWO_CH_HOSTS:
        raise ScraperError("URL должен указывать на публичную доску или тред 2ch.org")

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
        "Ожидался URL доски https://2ch.org/b/ или треда https://2ch.org/b/res/123.html"
    )


def clean_post_text(html: str) -> str:
    """Преобразовать HTML-фрагмент комментария API в читаемый plain text."""

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
    """Вернуть уникальные ссылки на посты в исходном формате ``>>123``."""

    return tuple(dict.fromkeys(f">>{post_id}" for post_id in REFERENCE_PATTERN.findall(text)))


def _require_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ScraperError(f"Некорректный ответ 2ch API: {context} должен быть объектом")
    return cast(dict[str, object], value)


def _require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ScraperError(f"Некорректный ответ 2ch API: {context} должен быть списком")
    return cast(list[object], value)


def _require_integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ScraperError(f"Некорректный ответ 2ch API: {context} должен быть целым числом")
    return value


def parse_catalog_thread_ids(payload: object, max_threads: int | None = None) -> list[int]:
    """Извлечь упорядоченные идентификаторы тредов из ответа catalog API."""

    if max_threads is not None and max_threads <= 0:
        raise ValueError("max_threads должен быть положительным")

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
    """Создать текстовый документ треда из ответа JSON API."""

    root = _require_object(payload, "thread root")
    raw_threads = _require_list(root.get("threads"), "threads")
    if not raw_threads:
        raise ScraperError("Некорректный ответ 2ch API: список тредов пуст")
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
        posts.append(PostRecord(post_id, text, extract_references(text)))
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
