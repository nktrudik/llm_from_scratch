"""Build a dialogue-style JSONL dataset from immutable raw 2ch threads."""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from bs4 import BeautifulSoup

from mini_llm.deduplication import ContentDeduplicator
from mini_llm.preprocessing_stats import PreprocessingStats

REFERENCE_PATTERN = re.compile(r">>(\d+)")
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


class PreprocessingError(RuntimeError):
    """Raised when a raw thread does not match the expected schema."""


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Paths and conservative filtering settings for preprocessing."""

    input_dir: Path = Path("data/raw/2ch")
    output_dir: Path = Path("data/processed")
    review_size: int = 100
    random_seed: int = 42
    dedup_min_characters: int = 80
    near_duplicate_threshold: float = 0.9

    def __post_init__(self) -> None:
        if self.review_size < 0:
            raise ValueError("review_size must be non-negative")
        if self.dedup_min_characters <= 0:
            raise ValueError("dedup_min_characters must be positive")
        if not 0.0 < self.near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be in the range (0, 1]")


@dataclass(frozen=True, slots=True)
class RawPost:
    """Post fields needed to reconstruct a reply graph."""

    post_id: int
    text: str
    references: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RawThread:
    """Validated immutable raw thread."""

    board: str
    thread_id: int
    source_path: str
    posts: tuple[RawPost, ...]


class _ReviewReservoir:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self._seen = 0
        self._items: list[dict[str, object]] = []
        # This PRNG is intentionally deterministic for a reproducible manual-review sample.
        self._random = random.Random(seed)  # nosec B311

    @property
    def items(self) -> list[dict[str, object]]:
        return self._items

    def consider(self, sample: dict[str, object]) -> None:
        self._seen += 1
        if self.size == 0:
            return
        if len(self._items) < self.size:
            self._items.append(sample)
            return
        replacement_index = self._random.randrange(self._seen)
        if replacement_index < self.size:
            self._items[replacement_index] = sample


def _require_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PreprocessingError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise PreprocessingError(f"{context} must be a list")
    return cast(list[object], value)


def _require_int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PreprocessingError(f"{context} must be an integer")
    return value


def load_raw_thread(path: Path, input_dir: Path) -> RawThread:
    """Read and validate one raw thread without modifying it."""

    try:
        payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreprocessingError(f"Cannot read {path}: {error}") from error
    root = _require_object(payload, str(path))
    board = root.get("board")
    if not isinstance(board, str) or not board:
        raise PreprocessingError(f"{path}: board must be a non-empty string")
    thread_id = _require_int(root.get("thread_id"), f"{path}: thread_id")

    posts: list[RawPost] = []
    for index, raw_post in enumerate(_require_list(root.get("posts"), f"{path}: posts")):
        post = _require_object(raw_post, f"{path}: posts[{index}]")
        post_id = _require_int(post.get("post_id"), f"{path}: posts[{index}].post_id")
        text = post.get("text")
        if not isinstance(text, str):
            raise PreprocessingError(f"{path}: posts[{index}].text must be a string")
        raw_references = _require_list(post.get("references"), f"{path}: posts[{index}].references")
        references = tuple(
            int(match.group(1))
            for value in raw_references
            if isinstance(value, str) and (match := REFERENCE_PATTERN.fullmatch(value))
        )
        posts.append(RawPost(post_id, text, tuple(dict.fromkeys(references))))

    return RawThread(board, thread_id, path.relative_to(input_dir).as_posix(), tuple(posts))


def clean_training_text(text: str) -> str:
    """Remove technical/identifying noise while preserving the original voice."""

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
    """Keep short dialogue turns when they contain a visible letter, number, or symbol."""

    return any(unicodedata.category(character)[0] in {"L", "N", "S"} for character in text)


def is_image_dependent(text: str) -> bool:
    """Detect short prompts that are meaningless without an omitted attachment."""

    compact = re.sub(r"\s+", " ", text).strip(" .,!?:;—-")
    return len(compact) <= 80 and IMAGE_DEPENDENT_PATTERN.fullmatch(compact) is not None


def _remove_context_quotes(response: str, context_texts: Sequence[str]) -> str:
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


def _split_multi_reference_response(text: str, parent_ids: Sequence[int]) -> dict[int, str] | None:
    if len(parent_ids) < 2:
        return None
    valid_parents = set(parent_ids)
    groups: dict[int, list[str]] = {}
    current_parent: int | None = None
    for line in (line.strip() for line in text.splitlines() if line.strip()):
        prefix_match = REFERENCE_PREFIX_PATTERN.match(line)
        if prefix_match:
            line_references = [
                int(value) for value in REFERENCE_PATTERN.findall(prefix_match["refs"])
            ]
            if len(line_references) != 1 or line_references[0] not in valid_parents:
                return None
            current_parent = line_references[0]
            groups.setdefault(current_parent, [])
            if prefix_match["body"].strip():
                groups[current_parent].append(prefix_match["body"].strip())
        elif current_parent is None:
            return None
        else:
            groups[current_parent].append(line)

    if set(groups) != valid_parents or any(not lines for lines in groups.values()):
        return None
    return {parent_id: "\n".join(lines) for parent_id, lines in groups.items()}


def _context_ids(
    parent_ids: Sequence[int], parents_by_id: dict[int, tuple[int, ...]], order: dict[int, int]
) -> list[int]:
    ancestors: set[int] = set()

    def visit(post_id: int) -> None:
        if post_id in ancestors:
            return
        for parent_id in parents_by_id.get(post_id, ()):
            visit(parent_id)
        ancestors.add(post_id)

    for parent_id in parent_ids:
        visit(parent_id)
    return sorted(ancestors, key=order.__getitem__)


def iter_thread_samples(
    thread: RawThread,
    deduplicator: ContentDeduplicator,
    stats: PreprocessingStats,
) -> Iterator[dict[str, object]]:
    """Yield training samples from one reply graph in source order."""

    if len(thread.posts) < 2:
        return
    order = {post.post_id: index for index, post in enumerate(thread.posts)}
    root_id = thread.posts[0].post_id
    parents_by_id: dict[int, tuple[int, ...]] = {root_id: ()}
    cleaned_by_id: dict[int, str] = {}
    invalid_by_id: dict[int, str] = {}

    for post in thread.posts:
        cleaned = clean_training_text(post.text)
        if not is_meaningful_text(cleaned):
            invalid_by_id[post.post_id] = "empty_after_cleaning"
        elif is_image_dependent(cleaned):
            invalid_by_id[post.post_id] = "image_dependent"
        else:
            cleaned_by_id[post.post_id] = cleaned

        valid_parents = tuple(
            reference
            for reference in post.references
            if reference in order and order[reference] < order[post.post_id]
        )
        if post.post_id != root_id:
            parents_by_id[post.post_id] = valid_parents

    for post in thread.posts[1:]:
        stats.candidate_responses += 1
        if post.post_id in invalid_by_id:
            stats.dropped[invalid_by_id[post.post_id]] += 1
            continue

        direct_parents = parents_by_id[post.post_id]
        if len(post.references) > 1:
            stats.multi_reference_posts += 1
        if not direct_parents:
            stats.dropped["missing_reliable_parent"] += 1
            if not post.references and root_id in cleaned_by_id:
                old_fallback_response = _remove_context_quotes(
                    cleaned_by_id[post.post_id], [cleaned_by_id[root_id]]
                )
                if is_meaningful_text(old_fallback_response) and not is_image_dependent(
                    old_fallback_response
                ):
                    stats.thread_root_fallback_samples_prevented += 1
            continue

        split_responses = _split_multi_reference_response(post.text, direct_parents)
        if split_responses is None:
            drafts = [(direct_parents, post.text, False)]
        else:
            drafts = [((parent_id,), text, True) for parent_id, text in split_responses.items()]

        for draft_parents, raw_response, was_split in drafts:
            if any(parent_id not in cleaned_by_id for parent_id in draft_parents):
                stats.dropped["missing_context"] += 1
                continue
            context_ids = _context_ids(draft_parents, parents_by_id, order)
            available_context_ids = [post_id for post_id in context_ids if post_id in cleaned_by_id]
            context = [
                {"post_id": post_id, "text": cleaned_by_id[post_id]}
                for post_id in available_context_ids
            ]
            if not context:
                stats.dropped["missing_context"] += 1
                continue

            response_text = clean_training_text(raw_response)
            response_text = _remove_context_quotes(
                response_text, [cast(str, message["text"]) for message in context]
            )
            if not is_meaningful_text(response_text):
                stats.dropped["empty_after_cleaning"] += 1
                continue
            if is_image_dependent(response_text):
                stats.dropped["image_dependent"] += 1
                continue

            duplicate_status = deduplicator.classify(response_text)
            if duplicate_status != "unique":
                stats.dropped[duplicate_status] += 1
                continue

            sample = {
                "board": thread.board,
                "thread_id": thread.thread_id,
                "source_post_ids": [*available_context_ids, post.post_id],
                "context": context,
                "response": {"post_id": post.post_id, "text": response_text},
                "metadata": {
                    "source_file": thread.source_path,
                    "referenced_post_ids": list(draft_parents),
                    "context_depth": len(context),
                    "multi_reference": len(direct_parents) > 1,
                    "split_from_multi_reference": was_split,
                    "parent_strategy": "explicit_references",
                },
            }
            stats.record_sample(sample)
            yield sample


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def preprocess_dataset(config: PreprocessingConfig | None = None) -> PreprocessingStats:
    """Stream raw threads into dataset, statistics, and review JSONL files."""

    active_config = config or PreprocessingConfig()
    if not active_config.input_dir.is_dir():
        raise PreprocessingError(f"Raw input directory does not exist: {active_config.input_dir}")
    active_config.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = active_config.output_dir / "2ch_dialogues.jsonl"
    review_path = active_config.output_dir / "2ch_review_sample.jsonl"
    stats_path = active_config.output_dir / "2ch_preprocessing_stats.json"
    temporary_dataset_path = dataset_path.with_suffix(".jsonl.tmp")
    stats = PreprocessingStats()
    deduplicator = ContentDeduplicator(
        minimum_characters=active_config.dedup_min_characters,
        similarity_threshold=active_config.near_duplicate_threshold,
    )
    review = _ReviewReservoir(active_config.review_size, active_config.random_seed)

    with temporary_dataset_path.open("w", encoding="utf-8", newline="\n") as dataset_file:
        for path in sorted(active_config.input_dir.glob("*/*.json")):
            stats.raw_threads_read += 1
            try:
                thread = load_raw_thread(path, active_config.input_dir)
            except PreprocessingError:
                stats.dropped["malformed_thread"] += 1
                continue
            stats.raw_posts_read += len(thread.posts)
            stats.boards[thread.board] += 1
            for sample in iter_thread_samples(thread, deduplicator, stats):
                dataset_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                review.consider(sample)
    temporary_dataset_path.replace(dataset_path)

    temporary_review_path = review_path.with_suffix(".jsonl.tmp")
    with temporary_review_path.open("w", encoding="utf-8", newline="\n") as review_file:
        for sample in review.items:
            review_file.write(json.dumps(sample, ensure_ascii=False, indent=None) + "\n")
    temporary_review_path.replace(review_path)
    stats.review_samples = len(review.items)
    _write_json(stats_path, stats.to_dict())
    return stats


def build_argument_parser() -> argparse.ArgumentParser:
    """Build preprocessing CLI arguments."""

    parser = argparse.ArgumentParser(description="Build dialogue JSONL from raw 2ch threads.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/2ch"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--review-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run preprocessing from the command line."""

    args = build_argument_parser().parse_args(argv)
    try:
        stats = preprocess_dataset(
            PreprocessingConfig(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                review_size=args.review_size,
                random_seed=args.seed,
            )
        )
    except (PreprocessingError, ValueError) as error:
        print(f"Preprocessing failed: {error}")
        return 1
    print(
        f"Created {stats.samples_created} samples from {stats.raw_threads_read} raw threads "
        f"in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
