"""Тесты BPE tokenizer и dialogue serialization."""

import json
from pathlib import Path

import pytest

from mini_llm.data.dialogue import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    USER_TOKEN,
    DialogueSample,
    serialize_dialogue,
)
from mini_llm.modeling import ModelConfig
from mini_llm.tokenization import BPETokenizer, train_bpe_tokenizer


def _write_train_split(path: Path) -> None:
    samples = [
        {
            "board": "b",
            "thread_id": 1,
            "context": [{"post_id": 1, "text": "Привет, как дела? 😎"}],
            "response": {"post_id": 2, "text": "Нормально! Mixed English 123."},
        },
        {
            "board": "b",
            "thread_id": 2,
            "context": [{"post_id": 3, "text": "Сленг и пунктуация?!"}],
            "response": {"post_id": 4, "text": "Да, всё остаётся как есть."},
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )


def _trained_tokenizer(tmp_path: Path) -> tuple[BPETokenizer, Path]:
    train_file = tmp_path / "train.jsonl"
    output_file = tmp_path / "tokenizer.json"
    _write_train_split(train_file)
    tokenizer = train_bpe_tokenizer(
        train_file,
        output_file,
        vocab_size=300,
        min_frequency=1,
        show_progress=False,
    )
    return tokenizer, output_file


def test_special_token_ids_match_model_config(tmp_path: Path) -> None:
    tokenizer, _ = _trained_tokenizer(tmp_path)
    config = ModelConfig()

    assert tokenizer.token_to_id(PAD_TOKEN) == config.pad_token_id
    assert tokenizer.token_to_id(UNK_TOKEN) == config.unk_token_id
    assert tokenizer.token_to_id(BOS_TOKEN) == config.bos_token_id
    assert tokenizer.token_to_id(EOS_TOKEN) == config.eos_token_id
    assert tokenizer.token_to_id(USER_TOKEN) == config.user_token_id
    assert tokenizer.token_to_id(ASSISTANT_TOKEN) == config.assistant_token_id


def test_encode_decode_round_trip_and_reload(tmp_path: Path) -> None:
    tokenizer, output_file = _trained_tokenizer(tmp_path)
    text = "Русский + English, 123; сленг ёжик 🤖\nновая строка"
    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text
    loaded = BPETokenizer.load(output_file)
    assert loaded.encode(text) == token_ids
    assert loaded.decode(token_ids) == text


def test_dialogue_format_and_window_keep_response(tmp_path: Path) -> None:
    tokenizer, _ = _trained_tokenizer(tmp_path)
    sample = DialogueSample(
        "b",
        10,
        ("Очень старый длинный контекст " * 10, "Последнее сообщение"),
        "Обязательный ответ",
    )

    serialized = serialize_dialogue(sample)
    assert serialized.startswith(BOS_TOKEN + ASSISTANT_TOKEN)
    assert USER_TOKEN + "Последнее сообщение" in serialized
    assert serialized.endswith(EOS_TOKEN)

    response_ids = tokenizer.encode_response(sample)
    max_length = len(response_ids) + 8
    window = tokenizer.encode_dialogue_window(sample, max_length=max_length)
    assert len(window) <= max_length
    assert window[-len(response_ids) :] == response_ids

    too_long = DialogueSample("b", 11, ("context",), "response " * 100)
    with pytest.raises(ValueError, match="Response"):
        tokenizer.encode_dialogue_window(too_long, max_length=10)
