"""Очищенные SFT-пары: verbatim текст, строгая схема, exact dedup и thread split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from mini_llm.api.schemas import PreprocessingRequest
from mini_llm.api.services import run_preprocessing
from mini_llm.data.dataset import DialogueDataset
from mini_llm.data.dialogue import DialogueSample, iter_dialogue_jsonl
from mini_llm.data.interfaces import EncodedDialogue
from mini_llm.data.preprocessing import PreprocessingConfig, PreprocessingError
from mini_llm.data.preprocessing.cli import main
from mini_llm.data.preprocessing.sft import SFTPreprocessingConfig, preprocess_sft_dataset
from mini_llm.data.splitting import SPLIT_NAMES


def _sample(
    thread_id: int, question: str, answer: str, *, self_contained: bool = True
) -> dict[str, object]:
    return {
        "sample_id": f"{thread_id}:1:2",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "board": "b",
            "thread_id": thread_id,
            "user_post_id": thread_id * 10 + 1,
            "assistant_post_id": thread_id * 10 + 2,
            "self_contained_prompt": self_contained,
            "ancestor_chain": [{"role": "assistant", "content": "Не добавлять"}],
        },
    }


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class TinyTokenizer:
    """Минимальный tokenizer для проверки совместимости выходного JSONL с Dataset."""

    pad_token_id = 0
    vocab_size = 16

    def encode_training_window(self, sample: DialogueSample, *, max_length: int) -> EncodedDialogue:
        assert len(sample.context) == 1
        return EncodedDialogue([1, 6, 7, 2], response_start=2)


def test_sft_keeps_text_metadata_and_single_user_context_verbatim(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    question = "  БЛЯТЬ\t CAPS 😎\r\nhttps://example.com a@b.com >>123\u200b  "
    answer = "\n Да!  е\u0301 ё\t\n"
    source_sample = _sample(1, question, answer)
    path = raw / "1_sft_raw.json"
    _write(path, {"samples": [source_sample]})
    original = path.read_bytes()
    # Combined и старые posts не являются источниками даже при невалидном JSON.
    (raw / "combined_any_sft_raw.json").write_text("{", encoding="utf-8")
    (raw / "1.json").write_text("{", encoding="utf-8")
    stats = preprocess_sft_dataset(SFTPreprocessingConfig(raw, output))
    records = list(iter_dialogue_jsonl(output / "2ch_dialogues.jsonl"))
    payload, dialogue = records[0]
    assert dialogue.context == (question,)
    assert dialogue.response == answer
    assert payload["context"] == [{"post_id": 11, "text": question}]
    assert payload["source_post_ids"] == [11, 12]
    assert payload["metadata"] == source_sample["metadata"]
    assert payload["sample_id"] == source_sample["sample_id"]
    assert stats.raw_samples_found == stats.final_samples == stats.self_contained_samples == 1
    assert stats.source_files == 1 and stats.malformed_files == 0
    assert path.read_bytes() == original
    review = json.loads((output / "2ch_qa_review.jsonl").read_text(encoding="utf-8"))
    assert review == {"question": question, "answer": answer, "board": "b", "thread_id": 1}
    assert len(DialogueDataset(output / "2ch_dialogues.jsonl", TinyTokenizer())) == 1


def test_sft_deduplicates_only_exact_question_answer_pairs(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    samples = [
        _sample(1, "Вопрос", "Ответ"),
        _sample(2, "Вопрос", "Ответ"),
        _sample(3, "Вопрос ", "Ответ"),
        _sample(4, "Вопрос", "Ответ "),
        _sample(5, "Вопрос", "Ответ!"),
        _sample(6, "Вопрос", "Другое"),
        _sample(7, "Не самодостаточный", "Ответ", self_contained=False),
    ]
    _write(raw / "1_sft_raw.json", samples)
    stats = preprocess_sft_dataset(SFTPreprocessingConfig(raw, output))
    assert stats.raw_samples_found == 7
    assert stats.self_contained_samples == 6
    assert stats.final_samples == 5
    assert stats.dropped == {"exact_duplicate": 1, "not_self_contained": 1}
    assert stats.samples_by_board == {"b": 5}


@pytest.mark.parametrize(
    "case",
    [
        "roles",
        "extra_message",
        "empty_user",
        "empty_assistant",
        "no_metadata",
        "bad_id",
        "bad_flag",
        "missing_flag",
        "missing_sample_id",
        "non_string",
        "not_object",
    ],
)
def test_sft_drops_malformed_samples_without_affecting_valid_one(tmp_path: Path, case: str) -> None:
    valid = _sample(1, "Вопрос", "Ответ")
    bad = _sample(2, "Другой вопрос", "Другой ответ")
    messages = cast(list[dict[str, object]], bad["messages"])
    metadata = cast(dict[str, object], bad["metadata"])
    malformed: object = bad
    if case == "roles":
        messages[0]["role"] = "assistant"
    elif case == "extra_message":
        messages.append({"role": "user", "content": "Третий"})
    elif case == "empty_user":
        messages[0]["content"] = " \n\t"
    elif case == "empty_assistant":
        messages[1]["content"] = ""
    elif case == "no_metadata":
        bad.pop("metadata")
    elif case == "bad_id":
        metadata["user_post_id"] = True
    elif case == "bad_flag":
        metadata["self_contained_prompt"] = 1
    elif case == "missing_flag":
        metadata.pop("self_contained_prompt")
    elif case == "missing_sample_id":
        bad.pop("sample_id")
    elif case == "non_string":
        messages[1]["content"] = 7
    else:
        malformed = None
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    _write(raw / "1_sft_raw.json", {"samples": [malformed, valid]})
    stats = preprocess_sft_dataset(SFTPreprocessingConfig(raw, output))
    assert stats.raw_samples_found == 2
    assert stats.dropped["malformed"] == stats.final_samples == 1


def test_sft_split_has_no_thread_leakage_and_review_is_reproducible(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for thread in range(20):
        _write(
            raw / f"{thread}_sft_raw.json",
            {
                "samples": [
                    _sample(thread, f"Вопрос {thread}", f"Ответ {thread}"),
                    _sample(thread, f"Вопрос {thread}", f"Второй ответ {thread}"),
                ]
            },
        )
    first, second = tmp_path / "first", tmp_path / "second"
    stats = preprocess_sft_dataset(SFTPreprocessingConfig(raw, first, review_size=10))
    preprocess_sft_dataset(SFTPreprocessingConfig(raw, second, review_size=10))
    assert (first / "2ch_qa_review.jsonl").read_bytes() == (
        second / "2ch_qa_review.jsonl"
    ).read_bytes()
    assert stats.review_samples == 10
    threads: dict[int, str] = {}
    for split in SPLIT_NAMES:
        path = Path("splits") / f"{split}.jsonl"
        assert (first / path).read_bytes() == (second / path).read_bytes()
        for _, sample in iter_dialogue_jsonl(first / path):
            assert threads.setdefault(sample.thread_id, split) == split
    splits = cast(dict[str, dict[str, object]], stats.split_report["splits"])
    assert [splits[name]["samples"] for name in SPLIT_NAMES] == [36, 2, 2]
    report = json.loads((first / "2ch_preprocessing_stats.json").read_text(encoding="utf-8"))
    assert report["final_samples"] == 40
    assert report["split_statistics"] == stats.split_report


def test_sft_refuses_to_replace_dataset_without_usable_sources(tmp_path: Path) -> None:
    raw, output = tmp_path / "raw", tmp_path / "processed"
    _write(
        raw / "1_sft_raw.json", {"samples": [_sample(1, "Вопрос", "Ответ", self_contained=False)]}
    )
    output.mkdir()
    path = output / "2ch_dialogues.jsonl"
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(PreprocessingError, match="не заменён"):
        preprocess_sft_dataset(SFTPreprocessingConfig(raw, output))
    assert path.read_text(encoding="utf-8") == "existing"


def test_sft_reports_malformed_files_and_never_uses_combined_sources(tmp_path: Path) -> None:
    raw, output = tmp_path / "raw", tmp_path / "processed"
    _write(raw / "combined_only_sft_raw.json", {"samples": [_sample(1, "Вопрос", "Ответ")]})
    with pytest.raises(PreprocessingError, match="Не найдены per-thread"):
        preprocess_sft_dataset(SFTPreprocessingConfig(raw, output))
    _write(raw / "2_sft_raw.json", _sample(2, "Второй вопрос", "Второй ответ"))
    (raw / "3_sft_raw.json").write_text("{", encoding="utf-8")
    stats = preprocess_sft_dataset(SFTPreprocessingConfig(raw, output))
    assert stats.source_files == 2
    assert stats.malformed_files == 1
    assert stats.raw_samples_found == stats.final_samples == 1


def test_sft_cli_and_api_select_new_source_without_changing_posts_defaults(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write(raw / "1_sft_raw.json", _sample(1, "Вопрос", "Ответ"))
    assert (
        main(
            [
                "--source-format",
                "sft",
                "--input-dir",
                str(raw),
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    request = PreprocessingRequest(source_format="sft", input_dir=raw, output_dir=tmp_path / "api")
    assert run_preprocessing(request)["final_samples"] == 1
    assert PreprocessingRequest().to_config() == PreprocessingConfig()
    assert PreprocessingRequest(source_format="sft").input_dir == Path("data/raw/2ch_clean")
