"""Построение обучающих samples из графа явных ответов."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

from mini_llm.deduplication import ContentDeduplicator
from mini_llm.preprocessing.cleaning import (
    REFERENCE_PREFIX_PATTERN,
    clean_training_text,
    is_image_dependent,
    is_meaningful_text,
    remove_context_quotes,
)
from mini_llm.preprocessing.parsing import REFERENCE_PATTERN
from mini_llm.preprocessing.schemas import RawThread
from mini_llm.preprocessing_stats import PreprocessingStats


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
    """Отдать samples одного графа ответов в исходном порядке."""

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
                old_fallback_response = remove_context_quotes(
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
            response_text = remove_context_quotes(
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
