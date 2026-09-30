"""Минимальная генерация через подготовленную pretrained-модель."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor

from mini_llm.pretrained.model import PreparedPretrained


@dataclass(frozen=True, slots=True)
class PretrainedGenerationResult:
    """Текст ответа и только новые token IDs."""

    text: str
    token_ids: list[int]


class _GenerativeModel(Protocol):
    def generate(self, **kwargs: object) -> object: ...


@torch.no_grad()
def generate_pretrained(
    prepared: PreparedPretrained,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> PretrainedGenerationResult:
    """Сгенерировать ответ без привязки к внутреннему классу Transformers."""

    if not prompt.strip():
        raise ValueError("prompt не может быть пустым")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens не может быть отрицательным")
    if temperature <= 0:
        raise ValueError("temperature должна быть положительной")
    prompt_ids = prepared.tokenizer.encode_prompt(prompt)
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
        eos_token_id=prepared.tokenizer.eos_token_id,
    )
    if not isinstance(generated, Tensor):
        raise RuntimeError("Метод generate не вернул Tensor")
    token_ids = cast(list[int], generated[0, len(prompt_ids) :].tolist())
    eos_id = prepared.tokenizer.eos_token_id
    if eos_id in token_ids:
        token_ids = token_ids[: token_ids.index(eos_id)]
    return PretrainedGenerationResult(
        prepared.tokenizer.decode(token_ids, skip_special_tokens=True),
        token_ids,
    )
