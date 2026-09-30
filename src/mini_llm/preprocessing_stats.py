"""Статистика построения обработанного dialogue dataset."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0, "p90": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
        "median": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, (len(ordered) * 9) // 10)],
    }


@dataclass(slots=True)
class PreprocessingStats:
    """Счётчики и распределения, собираемые при потоковой обработке samples."""

    raw_threads_read: int = 0
    raw_posts_read: int = 0
    candidate_responses: int = 0
    samples_created: int = 0
    multi_reference_posts: int = 0
    multi_reference_samples: int = 0
    split_multi_reference_samples: int = 0
    thread_root_fallback_samples_prevented: int = 0
    review_samples: int = 0
    boards: Counter[str] = field(default_factory=Counter)
    dropped: Counter[str] = field(default_factory=Counter)
    context_char_lengths: list[int] = field(default_factory=list)
    response_char_lengths: list[int] = field(default_factory=list)
    reply_chain_depths: list[int] = field(default_factory=list)

    def record_sample(self, sample: dict[str, object]) -> None:
        """Учесть один принятый sample и измерения его длины и глубины."""

        context = cast(list[dict[str, object]], sample["context"])
        response = cast(dict[str, object], sample["response"])
        metadata = cast(dict[str, object], sample["metadata"])
        self.samples_created += 1
        self.context_char_lengths.append(
            sum(len(cast(str, message["text"])) for message in context)
        )
        self.response_char_lengths.append(len(cast(str, response["text"])))
        self.reply_chain_depths.append(cast(int, metadata["context_depth"]))
        if cast(bool, metadata["multi_reference"]):
            self.multi_reference_samples += 1
        if cast(bool, metadata["split_from_multi_reference"]):
            self.split_multi_reference_samples += 1

    def to_dict(self) -> dict[str, object]:
        """Вернуть JSON-совместимый отчёт статистики."""

        duplicate_count = self.dropped["exact_duplicate"] + self.dropped["near_duplicate"]
        return {
            "raw_threads_read": self.raw_threads_read,
            "raw_posts_read": self.raw_posts_read,
            "candidate_responses": self.candidate_responses,
            "samples_created": self.samples_created,
            "samples_dropped": sum(self.dropped.values()),
            "dropped_by_reason": dict(sorted(self.dropped.items())),
            "duplicate_or_near_duplicate_samples": duplicate_count,
            "multi_reference_posts": self.multi_reference_posts,
            "multi_reference_samples": self.multi_reference_samples,
            "split_multi_reference_samples": self.split_multi_reference_samples,
            "messages_dropped_without_reliable_parent": self.dropped["missing_reliable_parent"],
            "thread_root_fallback_samples_prevented": (self.thread_root_fallback_samples_prevented),
            "review_samples": self.review_samples,
            "threads_by_board": dict(sorted(self.boards.items())),
            "context_character_lengths": _distribution(self.context_char_lengths),
            "response_character_lengths": _distribution(self.response_char_lengths),
            "reply_chain_depths": _distribution(self.reply_chain_depths),
        }
