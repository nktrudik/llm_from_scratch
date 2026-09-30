"""Тесты preprocessing диалогов и дедупликации содержимого."""

import json
from pathlib import Path
from typing import cast

from mini_llm.deduplication import ContentDeduplicator
from mini_llm.preprocessing import (
    PreprocessingConfig,
    RawPost,
    RawThread,
    clean_training_text,
    is_image_dependent,
    iter_thread_samples,
    preprocess_dataset,
)
from mini_llm.preprocessing_stats import PreprocessingStats


def _context_post_ids(sample: dict[str, object]) -> list[object]:
    context = cast(list[dict[str, object]], sample["context"])
    return [message["post_id"] for message in context]


def _nested_object(sample: dict[str, object], key: str) -> dict[str, object]:
    return cast(dict[str, object], sample[key])


def test_cleaning_removes_technical_noise_but_preserves_voice() -> None:
    source = (
        "МАТ CAPS сленг 😎\u200b https://example.com test@example.com "
        "+7 (999) 123-45-67 192.168.0.1 @telegram_name >>123 (OP)"
    )

    assert clean_training_text(source) == "МАТ CAPS сленг 😎"
    assert is_image_dependent("пикрил")
    assert not is_image_dependent("Да")


def test_reply_graph_requires_explicit_parent_and_keeps_ancestor_history() -> None:
    thread = RawThread(
        board="b",
        thread_id=10,
        source_path="b/10.json",
        posts=(
            RawPost(1, "Начальное сообщение задаёт тему разговора.", ()),
            RawPost(2, ">>1 Первый содержательный ответ по теме.", (1,)),
            RawPost(3, ">>2 Продолжение цепочки с новым аргументом.", (2,)),
            RawPost(4, ">>1 >>2 Общий ответ сразу двум предыдущим сообщениям.", (1, 2)),
            RawPost(5, "Да", ()),
        ),
    )
    stats = PreprocessingStats()
    deduplicator = ContentDeduplicator(minimum_characters=1_000)

    samples = list(iter_thread_samples(thread, deduplicator, stats))

    assert _context_post_ids(samples[0]) == [1]
    assert _context_post_ids(samples[1]) == [1, 2]
    assert _context_post_ids(samples[2]) == [1, 2]
    assert samples[1]["source_post_ids"] == [1, 2, 3]
    assert _nested_object(samples[2], "metadata")["multi_reference"] is True
    assert len(samples) == 3
    assert stats.dropped["missing_reliable_parent"] == 1
    assert stats.thread_root_fallback_samples_prevented == 1
    assert all(
        _nested_object(sample, "metadata")["parent_strategy"] == "explicit_references"
        for sample in samples
    )


def test_reliably_separated_multi_reference_response_creates_two_samples() -> None:
    thread = RawThread(
        board="b",
        thread_id=20,
        source_path="b/20.json",
        posts=(
            RawPost(1, "Первый родитель с самостоятельным текстом.", ()),
            RawPost(2, ">>1 Второй родитель продолжает обсуждение.", (1,)),
            RawPost(
                3,
                ">>1 Ответ только первому собеседнику.\n"
                ">>2 Отдельный ответ только второму собеседнику.",
                (1, 2),
            ),
        ),
    )
    stats = PreprocessingStats()

    samples = list(
        iter_thread_samples(thread, ContentDeduplicator(minimum_characters=1_000), stats)
    )
    split_samples = [
        sample for sample in samples if _nested_object(sample, "response")["post_id"] == 3
    ]

    assert len(split_samples) == 2
    assert [
        _nested_object(sample, "metadata")["referenced_post_ids"] for sample in split_samples
    ] == [[1], [2]]
    assert all(
        _nested_object(sample, "metadata")["split_from_multi_reference"] for sample in split_samples
    )


def test_deduplication_keeps_common_short_answers_and_detects_long_copies() -> None:
    deduplicator = ContentDeduplicator(minimum_characters=20, similarity_threshold=0.9)
    original_words = [f"слово{index}" for index in range(100)]
    original = " ".join(original_words)
    near_copy_words = original_words.copy()
    near_copy_words[50] = "изменение"

    assert deduplicator.classify("да") == "unique"
    assert deduplicator.classify("да") == "unique"
    assert deduplicator.classify(original) == "unique"
    assert deduplicator.classify(original) == "exact_duplicate"
    assert deduplicator.classify(" ".join(near_copy_words)) == "near_duplicate"


def _write_raw_thread(path: Path, thread_id: int, response: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "board": "b",
                "thread_id": thread_id,
                "posts": [
                    {
                        "post_id": thread_id,
                        "text": "Исходный вопрос для диалога.",
                        "references": [],
                    },
                    {
                        "post_id": thread_id + 1,
                        "text": response,
                        "references": [f">>{thread_id}"],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_pipeline_writes_dataset_stats_review_and_keeps_raw_immutable(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "2ch"
    processed_dir = tmp_path / "processed"
    repeated_response = (
        "Это достаточно длинный содержательный ответ, который повторяется в другом треде "
        "и поэтому должен остаться в итоговом наборе только один раз."
    )
    first_raw = raw_dir / "b" / "100.json"
    second_raw = raw_dir / "b" / "200.json"
    _write_raw_thread(first_raw, 100, repeated_response)
    _write_raw_thread(second_raw, 200, repeated_response)
    original_bytes = first_raw.read_bytes()

    stats = preprocess_dataset(
        PreprocessingConfig(
            input_dir=raw_dir,
            output_dir=processed_dir,
            review_size=10,
            dedup_min_characters=40,
        )
    )

    dataset_lines = (processed_dir / "2ch_dialogues.jsonl").read_text(encoding="utf-8").splitlines()
    stats_payload = cast(
        dict[str, object],
        json.loads((processed_dir / "2ch_preprocessing_stats.json").read_text(encoding="utf-8")),
    )
    review_lines = (
        (processed_dir / "2ch_review_sample.jsonl").read_text(encoding="utf-8").splitlines()
    )

    assert first_raw.read_bytes() == original_bytes
    assert len(dataset_lines) == 1
    assert len(review_lines) == 1
    assert stats.raw_threads_read == 2
    assert stats.samples_created == 1
    assert stats.dropped["exact_duplicate"] == 1
    assert stats_payload["duplicate_or_near_duplicate_samples"] == 1
    assert stats_payload["messages_dropped_without_reliable_parent"] == 0
    assert stats_payload["thread_root_fallback_samples_prevented"] == 0
