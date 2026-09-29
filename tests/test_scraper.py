"""Tests for HTML text extraction and raw-document serialization."""

import json
from pathlib import Path
from typing import cast

from mini_llm.scraper import ScrapedDocument, extract_document, save_document


def test_extracts_article_and_removes_page_chrome() -> None:
    html = """
    <html>
      <head><title>Readable Article</title><style>.hidden { display: none; }</style></head>
      <body>
        <header>Site navigation and account links should disappear completely.</header>
        <article>
          <h1>A useful article heading</h1>
          <p>This is the first substantial paragraph of the article. It contains enough detail to
          represent useful prose rather than a tiny label or an isolated interface element.</p>
          <p>The second paragraph adds more relevant information and makes the extracted document
          long enough for the configured minimum while preserving readable paragraph boundaries.</p>
        </article>
        <footer>Copyright, legal links, and other repetitive footer material.</footer>
        <script>window.trackingCode = "not useful text";</script>
      </body>
    </html>
    """

    document = extract_document(html, "https://example.com/article")

    assert document.title == "Readable Article"
    assert "first substantial paragraph" in document.text
    assert "second paragraph" in document.text
    assert "Site navigation" not in document.text
    assert "trackingCode" not in document.text
    assert "Copyright" not in document.text


def test_prefers_main_over_unrelated_large_div() -> None:
    html = """
    <html><body>
      <div>
        <p>This unrelated directory contains a large amount of text, but it is outside the main
        element and must not replace the explicitly marked primary content of this HTML page.</p>
        <p>Another unrelated paragraph makes this surrounding block deliberately longer.</p>
      </div>
      <main>
        <h1>Primary answer</h1>
        <p>The main element has semantic priority. This paragraph explains the important answer in
        a complete sentence with enough content to pass the individual block threshold.</p>
        <p>A follow-up paragraph provides additional context so that the selected main content is
        independently long enough to satisfy the document-level extraction threshold.</p>
      </main>
    </body></html>
    """

    document = extract_document(html, "https://example.com/answer", min_text_length=180)

    assert "semantic priority" in document.text
    assert "unrelated directory" not in document.text


def test_uses_semantic_block_heuristic_and_deduplicates_text() -> None:
    repeated = (
        "A detailed forum response can live inside an ordinary section without an article or main "
        "element, so the heuristic should still retain this useful discussion paragraph."
    )
    html = f"""
    <html><body>
      <div class="menu">
        <p>This menu description is intentionally long but still boilerplate.</p>
      </div>
      <section class="discussion-thread">
        <h2>Community explanation</h2>
        <p>{repeated}</p>
        <p>{repeated}</p>
        <ul>
          <li>A sufficiently descriptive list item should also be preserved as useful content.</li>
        </ul>
      </section>
      <div class="advertisement">
        <p>A long advertising message must be removed from output.</p>
      </div>
    </body></html>
    """

    document = extract_document(
        html,
        "https://forum.example.com/thread/1",
        min_text_length=150,
    )

    assert document.text.count(repeated) == 1
    assert "descriptive list item" in document.text
    assert "advertising message" not in document.text
    assert "menu description" not in document.text


def test_falls_back_to_body_paragraphs() -> None:
    html = """
    <html><body>
      <h1>Plain legacy page</h1>
      <p>A useful legacy page may place paragraphs directly under the body element. The fallback
      path should keep this prose even when semantic container elements are entirely absent.</p>
      <p>This second direct paragraph supplies additional material and verifies that the extractor
      returns clean paragraph breaks instead of collapsing the complete page into one line.</p>
    </body></html>
    """

    document = extract_document(html, "https://example.com/legacy", min_text_length=180)

    assert document.text.startswith("Plain legacy page")
    assert "\n\n" in document.text


def test_extracts_two_ch_post_messages_before_page_interface() -> None:
    html = """
    <html>
      <head><title>Тестовый тред — Двач</title></head>
      <body>
        <header>Доски Настройки Избранное</header>
        <main class="cntnt__main">
          <div class="thread" id="thread-123">
            <div class="post post_type_oppost">
              <div class="post__details">Аноним 29/09/26 Втр 12:00:00 №123</div>
              <article class="post__message post__message_op">
                Начальное сообщение обсуждения содержит полезный текст для будущего корпуса.
              </article>
              <button class="post__btn post__btn_type_menu">Меню сообщения</button>
            </div>
            <div class="post post_type_reply">
              <div class="post__details">Аноним 29/09/26 Втр 12:01:00 №124</div>
              <article class="post__message">
                Ответ продолжает тему и должен быть сохранён отдельным текстовым блоком.
              </article>
            </div>
          </div>
          <form class="postform">Форма ответа и служебные элементы интерфейса.</form>
        </main>
      </body>
    </html>
    """

    document = extract_document(
        html,
        "https://2ch.org/b/res/123.html",
        min_text_length=100,
    )

    assert document.title == "Тестовый тред — Двач"
    assert "Начальное сообщение" in document.text
    assert "Ответ продолжает тему" in document.text
    assert "Аноним 29/09/26" not in document.text
    assert "Меню сообщения" not in document.text
    assert "Форма ответа" not in document.text


def test_saves_document_as_json(tmp_path: Path) -> None:
    document = ScrapedDocument(
        source_url="https://example.com/posts/useful-page",
        title="Useful page",
        text="A sufficiently useful extracted document.",
        fetched_at="2026-09-29T12:00:00+00:00",
    )

    output_path = save_document(document, tmp_path)
    payload = cast(dict[str, str], json.loads(output_path.read_text(encoding="utf-8")))

    assert output_path.parent == tmp_path
    assert output_path.suffix == ".json"
    assert payload == document.to_dict()
