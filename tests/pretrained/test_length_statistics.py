"""Проверки статистики полных пар без сети, tokenizer assets и model weights."""

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from mini_llm.data.dialogue import DialogueSample
from mini_llm.data.interfaces import EncodedDialogue
from mini_llm.pretrained import PretrainedConfig, length_statistics


class LengthTokenizer:
    """Назначить каждому символу один токен и добавить десять служебных токенов."""

    vocab_size = 8192
    pad_token_id = 0

    def encode_training_window(self, sample: DialogueSample, *, max_length: int) -> EncodedDialogue:
        size = sum(map(len, sample.context)) + len(sample.response) + 10
        assert size <= max_length
        return EncodedDialogue(list(range(size)), size - len(sample.response))


def make_source(path: Path) -> None:
    """Записать пары с длинами точно на границах исследуемых окон и рядом с ними."""

    rows = [
        {
            "board": "b",
            "thread_id": index,
            "context": [{"text": "q" * (size - 12)}],
            "response": {"text": "ok"},
        }
        for index, size in enumerate((511, 512, 513, 768, 769, 1024, 1025))
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_pair_lengths_include_special_tokens_and_exact_window_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    make_source(path)
    original = path.read_bytes()
    report = length_statistics.analyze_pair_lengths(path, LengthTokenizer())
    assert report["samples"] == report["threads"] == 7
    assert report["full_pair_tokens"] == {
        "p50": 768,
        "p90": 1025,
        "p95": 1025,
        "p99": 1025,
        "max": 1025,
    }
    windows = cast(dict[str, dict[str, float]], report["fits_windows"])
    for window, count in (("512", 2), ("768", 4), ("1024", 6)):
        assert windows[window]["samples"] == count
        assert windows[window]["percentage"] == pytest.approx(100 * count / 7)
    assert report["sha256"] == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original


def test_empty_file_has_no_misleading_length_report(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="нет samples"):
        length_statistics.analyze_pair_lengths(path, LengthTokenizer())


def test_length_cli_writes_separate_report_without_changing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    make_source(source)
    config = tmp_path / "config.json"
    PretrainedConfig(model_id="Qwen/Qwen3-0.6B").save(config)
    monkeypatch.setattr(length_statistics, "load_local_tokenizer", lambda config: LengthTokenizer())
    output = tmp_path / "report.json"
    arguments = ["--pretrained-config", str(config), "--input-files", str(source), "--output-file"]
    original = source.read_bytes()
    assert length_statistics.main([*arguments, str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["enable_thinking"] is False
    assert report["datasets"][0]["samples"] == 7
    assert source.read_bytes() == original
    assert length_statistics.main([*arguments, str(source)]) == 1
    assert source.read_bytes() == original
