"""Подготовка очищенных SFT-пар без изменения текста сообщений."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from mini_llm.data.dialogue import parse_dialogue_sample, write_json
from mini_llm.data.preprocessing.pipeline import _ReviewReservoir
from mini_llm.data.preprocessing.schemas import PreprocessingError
from mini_llm.data.splitting import SplitConfig, split_dataset
from mini_llm.observability import terminal_log


@dataclass(frozen=True, slots=True)
class SFTPreprocessingConfig:
    """Настройки преобразования per-thread SFT JSON и последующего split."""

    input_dir: Path = Path("data/raw/2ch_clean")
    output_dir: Path = Path("data/processed")
    review_size: int = 100
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.review_size < 0:
            raise ValueError("review_size не может быть отрицательным")


@dataclass(slots=True)
class SFTPreprocessingStats:
    """Непересекающиеся причины удаления и итоговые размеры датасета."""

    source_files: int = 0
    malformed_files: int = 0
    raw_samples_found: int = 0
    self_contained_samples: int = 0
    final_samples: int = 0
    review_samples: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    samples_by_board: Counter[str] = field(default_factory=Counter)
    split_report: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Вернуть JSON-совместимую статистику SFT preprocessing и split."""

        return {
            "source_format": "sft",
            "source_files": self.source_files,
            "malformed_files": self.malformed_files,
            "raw_samples_found": self.raw_samples_found,
            "self_contained_samples": self.self_contained_samples,
            "malformed_samples_removed": self.dropped["malformed"],
            "duplicates_removed": self.dropped["exact_duplicate"],
            "non_self_contained_removed": self.dropped["not_self_contained"],
            "dropped_by_reason": dict(sorted(self.dropped.items())),
            "final_samples": self.final_samples,
            "review_samples": self.review_samples,
            "samples_by_board": dict(sorted(self.samples_by_board.items())),
            "split_statistics": self.split_report,
        }


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Ожидался JSON object")
    return cast(dict[str, object], value)


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Ожидался целочисленный ID")
    return value


def _load_samples(path: Path) -> list[object]:
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if isinstance(payload, list):
        return cast(list[object], payload)
    root = _object(payload)
    if isinstance(root.get("samples"), list):
        return cast(list[object], root["samples"])
    if "messages" in root:
        return [root]
    raise ValueError("В per-thread файле отсутствует samples")


def _convert_sample(root: dict[str, object], metadata: dict[str, object]) -> dict[str, object]:
    sample_id = root.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("Отсутствует sample_id")
    messages = root.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("Нужна пара ровно из двух messages")
    user = _object(messages[0])
    assistant = _object(messages[1])
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError("Нужны роли user -> assistant")
    question = user.get("content")
    answer = assistant.get("content")
    # strip используется только для проверки пустоты; исходные строки сохраняются целиком.
    if (
        not isinstance(question, str)
        or not question.strip()
        or not isinstance(answer, str)
        or not answer.strip()
    ):
        raise ValueError("Нужны непустые тексты user и assistant")
    user_id = _integer(metadata.get("user_post_id"))
    assistant_id = _integer(metadata.get("assistant_post_id"))
    sample: dict[str, object] = {
        "sample_id": sample_id,
        "board": metadata.get("board"),
        "thread_id": _integer(metadata.get("thread_id")),
        "source_post_ids": [user_id, assistant_id],
        "context": [{"post_id": user_id, "text": question}],
        "response": {"post_id": assistant_id, "text": answer},
        "metadata": metadata,
    }
    return sample


def preprocess_sft_dataset(config: SFTPreprocessingConfig | None = None) -> SFTPreprocessingStats:
    """Сохранить verbatim пары, exact-dedup, QA review и существующий thread-level split."""

    active = config or SFTPreprocessingConfig()
    if not active.input_dir.is_dir():
        raise PreprocessingError(f"Каталог raw-данных не существует: {active.input_dir}")
    paths = sorted(
        path
        for path in active.input_dir.rglob("*_sft_raw.json")
        if not path.name.lower().startswith("combined_") and path.is_file()
    )
    if not paths:
        raise PreprocessingError("Не найдены per-thread *_sft_raw.json; combined_* не используются")
    active.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = active.output_dir / "2ch_dialogues.jsonl"
    temporary = dataset_path.with_suffix(".jsonl.tmp")
    review = _ReviewReservoir(active.review_size, active.random_seed)
    seen_pairs: set[tuple[str, str]] = set()
    stats = SFTPreprocessingStats()
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for index, path in enumerate(paths, start=1):
            stats.source_files += 1
            try:
                raw_samples = _load_samples(path)
            except (OSError, UnicodeDecodeError, ValueError):
                stats.malformed_files += 1
                terminal_log("SFT", f"Пропущен malformed файл: {path}")
                continue
            for raw_sample in raw_samples:
                stats.raw_samples_found += 1
                try:
                    root = _object(raw_sample)
                    metadata = _object(root.get("metadata"))
                    flag = metadata.get("self_contained_prompt")
                    if not isinstance(flag, bool):
                        raise ValueError("self_contained_prompt должен быть boolean")
                    if not flag:
                        stats.dropped["not_self_contained"] += 1
                        continue
                    stats.self_contained_samples += 1
                    sample = _convert_sample(root, metadata)
                    dialogue = parse_dialogue_sample(sample)
                except ValueError:
                    stats.dropped["malformed"] += 1
                    continue
                pair = (dialogue.context[0], dialogue.response)
                if pair in seen_pairs:
                    stats.dropped["exact_duplicate"] += 1
                    continue
                seen_pairs.add(pair)
                output.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats.final_samples += 1
                stats.samples_by_board[dialogue.board] += 1
                review.consider(
                    {
                        "question": pair[0],
                        "answer": pair[1],
                        "board": dialogue.board,
                        "thread_id": dialogue.thread_id,
                    }
                )
            terminal_log(
                "SFT",
                f"Файл {index}/{len(paths)}: {path.name} "
                f"raw_samples={stats.raw_samples_found} accepted={stats.final_samples}",
            )
    if not stats.final_samples:
        raise PreprocessingError(
            "Нет пригодных self-contained пар; существующий dataset не заменён"
        )
    temporary.replace(dataset_path)
    review_path = active.output_dir / "2ch_qa_review.jsonl"
    temporary_review = review_path.with_suffix(".jsonl.tmp")
    with temporary_review.open("w", encoding="utf-8", newline="\n") as output:
        for sample in review.items:
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")
    temporary_review.replace(review_path)
    stats.review_samples = len(review.items)
    terminal_log("SFT", "Preprocessing завершён; выполняется thread-level split")
    stats.split_report = split_dataset(
        SplitConfig(dataset_path, active.output_dir / "splits", active.random_seed)
    )
    write_json(active.output_dir / "2ch_preprocessing_stats.json", stats.to_dict())
    return stats
