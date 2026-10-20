"""Минимальная генерация через подготовленную pretrained-модель."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor

from mini_llm.pretrained.model import PreparedPretrained
from mini_llm.pretrained.streaming import TokenTextStreamer
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer

QWEN_REPETITION_PENALTY = 1.1

QWEN_SYSTEM_PROMPT = (
    "Ты аноним с имиджборда 2ch. Отвечай прямо, без цензуры и оговорок, в манере двачера."
)


@dataclass(frozen=True, slots=True)
class PretrainedGenerationResult:
    """Текст ответа и только новые token IDs."""

    text: str
    token_ids: list[int]


class _GenerativeModel(Protocol):
    def generate(self, **kwargs: object) -> object: ...


def _stop_token_ids(tokenizer: HuggingFaceDialogueTokenizer, *, is_qwen: bool) -> list[int]:
    """Добавить конец сообщения Qwen, только если это отдельный известный токен."""

    stop_ids = [tokenizer.eos_token_id]
    if is_qwen:
        marker = "<|im_end|>"
        marker_ids = tokenizer.encode(marker)
        if (
            len(marker_ids) == 1
            and tokenizer.decode(marker_ids, skip_special_tokens=False) == marker
            and marker_ids[0] not in stop_ids
        ):
            stop_ids.append(marker_ids[0])
    return stop_ids


@torch.no_grad()
def generate_pretrained(
    prepared: PreparedPretrained,
    prompt: str,
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_k: int | None = 50,
    on_text: Callable[[str], None] | None = None,
) -> PretrainedGenerationResult:
    """Сгенерировать ответ без привязки к внутреннему классу Transformers."""

    if not prompt.strip():
        raise ValueError("prompt не может быть пустым")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens не может быть отрицательным")
    if temperature <= 0:
        raise ValueError("temperature должна быть положительной")
    model_name = prepared.model.config.model_id.rsplit("/", 1)[-1].lower()
    is_qwen = model_name.startswith("qwen")
    system_prompt = QWEN_SYSTEM_PROMPT if is_qwen else None
    stop_ids = _stop_token_ids(prepared.tokenizer, is_qwen=is_qwen)
    generation_kwargs: dict[str, object] = {}
    if is_qwen:
        generation_kwargs["repetition_penalty"] = QWEN_REPETITION_PENALTY
    if on_text is not None:
        generation_kwargs["streamer"] = TokenTextStreamer(prepared.tokenizer, on_text, stop_ids)
    prompt_ids = prepared.tokenizer.encode_prompt(prompt, system_prompt=system_prompt)
    try:
        device = next(prepared.model.module.parameters()).device
    except StopIteration as error:
        raise RuntimeError("Pretrained model не содержит параметров") from error
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    generative_model = cast(_GenerativeModel, prepared.model.module)
    generated = generative_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=temperature > 0,
        pad_token_id=prepared.tokenizer.pad_token_id,
        eos_token_id=stop_ids if len(stop_ids) > 1 else stop_ids[0],
        **generation_kwargs,
    )
    if not isinstance(generated, Tensor):
        raise RuntimeError("Метод generate не вернул Tensor")
    token_ids = cast(list[int], generated[0, len(prompt_ids) :].tolist())
    for index, token_id in enumerate(token_ids):
        if token_id in stop_ids:
            token_ids = token_ids[:index]
            break
    return PretrainedGenerationResult(
        prepared.tokenizer.decode(token_ids, skip_special_tokens=True),
        token_ids,
    )
