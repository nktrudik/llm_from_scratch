"""Очистка текста без стилистической нормализации сообщений."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Sequence

from bs4 import BeautifulSoup

from mini_llm.data.preprocessing.parsing import REFERENCE_PATTERN

REFERENCE_PREFIX_PATTERN = re.compile(r"^(?P<refs>(?:\s*>>\d+)+)\s*(?P<body>.*)$")
URL_PATTERN = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)")
HANDLE_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]{5,}\b")
OP_MARKER_PATTERN = re.compile(r"\(\s*OP\s*\)", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][^>]*>")
IMAGE_DEPENDENT_PATTERN = re.compile(
    r"^(?:пикрил(?:ейтед)?|вот (?:это|эта|этот|оно)|согласны\??|как вам\??|"
    r"что думаете\??|(?:смотри|см\.)?\s*(?:фото|картинк[ауе]|скрин))$",
    re.IGNORECASE,
)


def clean_training_text(text: str) -> str:
    """Удалить технический и идентифицирующий шум, сохранив голос автора."""

    cleaned = html.unescape(unicodedata.normalize("NFC", text))
    if HTML_TAG_PATTERN.search(cleaned):
        cleaned = BeautifulSoup(cleaned, "html.parser").get_text(" ")
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = EMAIL_PATTERN.sub(" ", cleaned)
    cleaned = IP_PATTERN.sub(" ", cleaned)
    cleaned = PHONE_PATTERN.sub(" ", cleaned)
    cleaned = HANDLE_PATTERN.sub(" ", cleaned)
    cleaned = REFERENCE_PATTERN.sub(" ", cleaned)
    cleaned = OP_MARKER_PATTERN.sub(" ", cleaned)
    cleaned = "".join(
        character
        for character in cleaned
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
    )
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def is_meaningful_text(text: str) -> bool:
    """Сохранить короткую реплику при наличии видимой буквы, цифры или символа."""

    return any(unicodedata.category(character)[0] in {"L", "N", "S"} for character in text)


def is_image_dependent(text: str) -> bool:
    """Найти короткую реплику, бессмысленную без удалённого вложения."""

    compact = re.sub(r"\s+", " ", text).strip(" .,!?:;—-")
    return len(compact) <= 80 and IMAGE_DEPENDENT_PATTERN.fullmatch(compact) is not None


def remove_context_quotes(response: str, context_texts: Sequence[str]) -> str:
    """Удалить из ответа дословные цитаты уже сохранённого контекста."""

    context_lines = {
        re.sub(r"\s+", " ", line).casefold()
        for context in context_texts
        for line in context.splitlines()
        if len(line.strip()) >= 12
    }
    kept_lines: list[str] = []
    for line in response.splitlines():
        comparison = re.sub(r"\s+", " ", line.lstrip("> ")).strip().casefold()
        if len(comparison) >= 12 and comparison in context_lines:
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()
