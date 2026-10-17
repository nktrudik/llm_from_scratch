"""Проверки родного instruct-формата и response-only границы без Hugging Face."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from mini_llm.data.dataset import IGNORE_INDEX, DialogueDataset, collate_causal_lm_batch
from mini_llm.data.dialogue import DialogueSample
from mini_llm.data.interfaces import OversizedResponseError
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer


class ChatTokenizer:
    """Маленький детерминированный tokenizer с явными role/end маркерами."""

    chat_template: str | dict[str, str] | None = "native-template"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    eos_token = "<end>"
    pad_token = "<pad>"

    def __len__(self) -> int:
        return 65536

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(char) + 100 for char in text]

    def decode(self, tokens: Sequence[int], *, skip_special_tokens: bool) -> str:
        return "".join(chr(token - 100) for token in tokens if token >= 100)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
    ) -> list[int]:
        assert tokenize
        assert not return_dict
        tokens: list[int] = []
        for message in messages:
            tokens.append(11 if message["role"] == "user" else 12)
            tokens.extend(self.encode(message["content"], add_special_tokens=False))
            tokens.append(self.eos_token_id)
        if add_generation_prompt:
            tokens.append(12)
        return tokens


def test_instruct_prompt_uses_native_chat_template() -> None:
    tokenizer = HuggingFaceDialogueTokenizer(ChatTokenizer())
    assert tokenizer.encode_prompt("Привет") == [11, *tokenizer.encode("Привет"), 2, 12]


def test_chat_window_preserves_response_and_masks_context() -> None:
    tokenizer = HuggingFaceDialogueTokenizer(ChatTokenizer())
    sample = DialogueSample("b", 1, ("старый" * 20, "ответ в context", "новый"), "response")
    window = tokenizer.encode_training_window(sample, max_length=22)
    prompt = tokenizer.encode_prompt("новый")
    assert window.token_ids[: window.response_start] == prompt
    assert window.token_ids[window.response_start :] == [*tokenizer.encode("response"), 2]
    assert len(window.token_ids) <= 22
    assert "старый" not in tokenizer.decode(window.token_ids)


def test_chat_window_does_not_truncate_oversized_response() -> None:
    tokenizer = HuggingFaceDialogueTokenizer(ChatTokenizer())
    with pytest.raises(OversizedResponseError):
        tokenizer.encode_training_window(
            DialogueSample("b", 1, ("context",), "r" * 40), max_length=16
        )


def test_tokenizer_without_chat_template_keeps_legacy_prompt() -> None:
    backend = ChatTokenizer()
    backend.chat_template = None
    tokenizer = HuggingFaceDialogueTokenizer(backend)
    assert tokenizer.encode_prompt("text") == [
        1,
        *tokenizer.encode("<USER>"),
        *tokenizer.encode("text"),
        *tokenizer.encode("<ASSISTANT>"),
    ]


class ThinkingTokenizer(ChatTokenizer):
    """Имитировать переключатель reasoning и записывать параметры шаблона."""

    def __init__(self) -> None:
        self.chat_template = "native-template-with-enable_thinking"
        self.calls: list[tuple[bool, bool | None]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        self.calls.append((add_generation_prompt, enable_thinking))
        tokens = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
        )
        if add_generation_prompt and enable_thinking is False:
            # Пустой think-блок входит в prompt, а не расходует бюджет генерации.
            tokens.extend([70, 71])
        elif not add_generation_prompt and enable_thinking is False:
            response_tokens = self.encode(messages[-1]["content"], add_special_tokens=False)
            response_start = len(tokens) - len(response_tokens) - 1
            tokens[response_start:response_start] = [70, 71]
        return tokens


@pytest.mark.parametrize(
    "template",
    ["native-enable_thinking", {"default": "native-enable_thinking"}],
)
def test_reasoning_is_disabled_in_inference_prompt(template: str | dict[str, str]) -> None:
    backend = ThinkingTokenizer()
    backend.chat_template = template
    tokenizer = HuggingFaceDialogueTokenizer(backend)
    assert tokenizer.encode_prompt("Привет") == [11, *tokenizer.encode("Привет"), 2, 12, 70, 71]
    assert backend.calls == [(True, False)]


def test_training_and_inference_use_the_same_non_thinking_prefix() -> None:
    backend = ThinkingTokenizer()
    tokenizer = HuggingFaceDialogueTokenizer(backend)
    tokenizer.encode_prompt("Привет")
    backend.calls.clear()
    sample = DialogueSample("b", 1, ("question",), "answer")
    window = tokenizer.encode_training_window(sample, max_length=64)
    assert backend.calls == [(True, False), (False, False)]
    assert window.token_ids[: window.response_start] == tokenizer.encode_prompt("question")
    assert window.token_ids[window.response_start :] == [*tokenizer.encode("answer"), 2]
    assert window.token_ids[window.response_start - 2 : window.response_start] == [70, 71]


def test_unused_reasoning_template_does_not_change_default_template() -> None:
    backend = ThinkingTokenizer()
    backend.chat_template = {"default": "native-template", "tool_use": "enable_thinking"}
    tokenizer = HuggingFaceDialogueTokenizer(backend)
    assert tokenizer.encode_prompt("text") == [11, *tokenizer.encode("text"), 2, 12]
    assert backend.calls == [(True, None)]


@pytest.mark.parametrize("native_template", [True, False])
def test_long_question_is_not_removed_to_fit_response(native_template: bool) -> None:
    backend = ChatTokenizer()
    if not native_template:
        backend.chat_template = None
    tokenizer = HuggingFaceDialogueTokenizer(backend)
    sample = DialogueSample("b", 1, ("question" * 20,), "ok")
    with pytest.raises(OversizedResponseError, match="Полная пара") as error:
        tokenizer.encode_training_window(sample, max_length=32)
    assert error.value.required_tokens > 32


def test_exact_pair_limit_keeps_the_entire_question_and_response() -> None:
    tokenizer = HuggingFaceDialogueTokenizer(ThinkingTokenizer())
    sample = DialogueSample("b", 1, ("  original question\n",), "answer")
    full = tokenizer.encode_training_window(sample, max_length=1024)
    assert tokenizer.encode_training_window(sample, max_length=len(full.token_ids)) == full
    assert full.token_ids[: full.response_start] == tokenizer.encode_prompt(sample.context[0])
    with pytest.raises(OversizedResponseError) as error:
        tokenizer.encode_training_window(sample, max_length=len(full.token_ids) - 1)
    assert error.value.required_tokens == len(full.token_ids)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_dataset_skips_long_pairs_and_masks_non_thinking_prompt(tmp_path: Path, split: str) -> None:
    samples = [
        {
            "board": "b",
            "thread_id": index,
            "context": [{"post_id": index * 2, "text": question}],
            "response": {"post_id": index * 2 + 1, "text": response},
        }
        for index, (question, response) in enumerate(
            [("original question", "answer"), ("q" * 100, "ok"), ("short", "a")]
        )
    ]
    path = tmp_path / f"{split}.jsonl"
    path.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    original = path.read_bytes()
    tokenizer = HuggingFaceDialogueTokenizer(ThinkingTokenizer())
    dataset = DialogueDataset(path, tokenizer, max_sequence_length=40)
    assert len(dataset) == 2
    assert dataset.scanned_samples == 3
    assert dataset.oversized_response_count == 1
    assert path.read_bytes() == original
    item = dataset[0]
    prompt_ids = tokenizer.encode_prompt("original question")
    assert item["input_ids"][: len(prompt_ids)].tolist() == prompt_ids
    labels = item["labels"].tolist()
    assert labels[: len(prompt_ids) - 1] == [IGNORE_INDEX] * (len(prompt_ids) - 1)
    assert labels[len(prompt_ids) - 1 :] == [*tokenizer.encode("answer"), 2]
    assert 70 not in labels and 71 not in labels
    batch = collate_causal_lm_batch([dataset[0], dataset[1]], pad_token_id=0)
    assert (batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX).all()
