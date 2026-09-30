"""Публичный интерфейс последовательного JSON-сборщика 2ch."""

from mini_llm.data.scraping.client import TwoChScraper
from mini_llm.data.scraping.parsing import (
    clean_post_text,
    extract_references,
    parse_catalog_thread_ids,
    parse_thread_document,
    parse_two_ch_url,
)
from mini_llm.data.scraping.schemas import (
    PostRecord,
    ScraperConfig,
    ScraperError,
    ThreadDocument,
    TwoChTarget,
)
from mini_llm.data.scraping.storage import save_thread_document, thread_output_path

__all__ = [
    "PostRecord",
    "ScraperConfig",
    "ScraperError",
    "ThreadDocument",
    "TwoChScraper",
    "TwoChTarget",
    "clean_post_text",
    "extract_references",
    "parse_catalog_thread_ids",
    "parse_thread_document",
    "parse_two_ch_url",
    "save_thread_document",
    "thread_output_path",
]
