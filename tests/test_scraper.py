"""Tests for collecting text-only threads from the 2ch JSON API."""

import json
from pathlib import Path
from typing import cast

import pytest

from mini_llm.scraper import (
    ScraperConfig,
    ScraperError,
    TwoChScraper,
    clean_post_text,
    extract_references,
    parse_catalog_thread_ids,
    parse_thread_document,
    parse_two_ch_url,
)


def thread_payload(thread_id: int, text: str = "Текст поста") -> dict[str, object]:
    """Return a minimal API response for one thread."""

    return {
        "title": "Название доски",
        "threads": [
            {
                "posts": [
                    {
                        "num": thread_id,
                        "comment": text,
                        "subject": "Тема треда",
                        "files": [{"path": "/ignored/image.jpg"}],
                    }
                ]
            }
        ],
    }


def test_cleans_html_and_extracts_unique_references() -> None:
    html = (
        '<p>Короткий ответ <a href="#123">&gt;&gt;123</a><br>'
        "следующая строка и <b>&gt;&gt;456</b>, снова &gt;&gt;123.</p>"
        "<script>unwanted()</script>"
    )

    text = clean_post_text(html)

    assert text == "Короткий ответ >>123\nследующая строка и >>456 , снова >>123."
    assert extract_references(text) == (">>123", ">>456")
    assert "unwanted" not in text


def test_parses_all_non_empty_posts_without_minimum_length() -> None:
    payload = {
        "title": "Доска",
        "threads": [
            {
                "posts": [
                    {"num": 100, "comment": "ОП", "subject": "Тема", "files": [{"name": "x"}]},
                    {"num": 101, "comment": "<br>   ", "subject": ""},
                    {"num": 102, "comment": "Ответ на <a>&gt;&gt;100</a>", "subject": ""},
                ]
            }
        ],
    }

    document = parse_thread_document(payload, "b", 100)
    serialized = document.to_dict()

    assert document.title == "Тема"
    assert [post.post_id for post in document.posts] == [100, 102]
    assert document.posts[0].text == "ОП"
    assert document.posts[1].references == (">>100",)
    assert "files" not in json.dumps(serialized)


def test_parses_thread_and_board_urls() -> None:
    thread_target = parse_two_ch_url("https://2ch.org/b/res/123456.html")
    board_target = parse_two_ch_url("https://2ch.org/po/")

    assert thread_target.board == "b"
    assert thread_target.thread_id == 123456
    assert board_target.board == "po"
    assert board_target.thread_id is None

    with pytest.raises(ScraperError):
        parse_two_ch_url("https://example.com/b/res/123456.html")
    with pytest.raises(ScraperError):
        parse_two_ch_url("https://2ch.org/b/catalog.html")


def test_catalog_preserves_order_deduplicates_and_applies_limit() -> None:
    payload = {"threads": [{"num": 30}, {"num": 20}, {"num": 30}, {"num": 10}]}

    assert parse_catalog_thread_ids(payload) == [30, 20, 10]
    assert parse_catalog_thread_ids(payload, max_threads=2) == [30, 20]


class StubTwoChScraper(TwoChScraper):
    """Network-free scraper returning predefined endpoint payloads."""

    def __init__(self, config: ScraperConfig, responses: dict[str, object]) -> None:
        super().__init__(config)
        self.responses = responses
        self.requested_urls: list[str] = []

    def fetch_json(self, url: str) -> object:
        self.requested_urls.append(url)
        return self.responses[url]


def test_board_url_downloads_limited_threads_sequentially(tmp_path: Path) -> None:
    catalog_url = "https://2ch.org/b/catalog.json"
    first_thread_url = "https://2ch.org/b/res/200.json"
    second_thread_url = "https://2ch.org/b/res/100.json"
    responses: dict[str, object] = {
        catalog_url: {"threads": [{"num": 200}, {"num": 100}]},
        first_thread_url: thread_payload(200),
        second_thread_url: thread_payload(100),
    }
    config = ScraperConfig(output_dir=tmp_path)

    with StubTwoChScraper(config, responses) as scraper:
        output_paths = scraper.scrape_to_files("https://2ch.org/b/", max_threads=1)

    assert scraper.requested_urls == [catalog_url, first_thread_url]
    assert output_paths == [tmp_path / "2ch" / "b" / "200.json"]
    payload = cast(dict[str, object], json.loads(output_paths[0].read_text(encoding="utf-8")))
    assert payload["thread_id"] == 200
    assert isinstance(payload["posts"], list)


def test_thread_url_uses_json_endpoint_without_catalog(tmp_path: Path) -> None:
    api_url = "https://2ch.org/b/res/777.json"
    config = ScraperConfig(output_dir=tmp_path)

    with StubTwoChScraper(config, {api_url: thread_payload(777, "Да")}) as scraper:
        output_paths = scraper.scrape_to_files("https://2ch.org/b/res/777.html")

    assert scraper.requested_urls == [api_url]
    assert output_paths[0].name == "777.json"
    assert output_paths[0].parent == tmp_path / "2ch" / "b"
