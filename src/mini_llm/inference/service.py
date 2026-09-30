"""Загрузка checkpoint и генерация текстового ответа по одному prompt."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from mini_llm.data.dialogue import ASSISTANT_TOKEN, BOS_TOKEN, EOS_TOKEN, USER_TOKEN
from mini_llm.inference.config import GenerationConfig
from mini_llm.inference.schemas import GenerationResult
from mini_llm.modeling import DecoderOnlyTransformer, ModelConfig
from mini_llm.tokenization import BPETokenizer


def _load_model(checkpoint_file: Path, device: torch.device) -> DecoderOnlyTransformer:
    if not checkpoint_file.is_file():
        raise RuntimeError(f"Checkpoint не найден: {checkpoint_file}")
    try:
        payload_object = cast(
            object,
            torch.load(checkpoint_file, map_location="cpu", weights_only=True),
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"Не удалось загрузить checkpoint {checkpoint_file}: {error}") from error
    if not isinstance(payload_object, dict):
        raise RuntimeError("Корень checkpoint должен быть объектом")
    payload = cast(dict[str, object], payload_object)
    saved_config = payload.get("model_config")
    model_config = ModelConfig()
    if not isinstance(saved_config, dict) or saved_config != asdict(model_config):
        raise RuntimeError("ModelConfig checkpoint не совпадает с текущей архитектурой")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Checkpoint не содержит model_state_dict")
    model = DecoderOnlyTransformer(model_config)
    model.load_state_dict(cast(Mapping[str, Tensor], state))
    model.to(device)
    model.eval()
    return model


def generate_response(prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
    """Сгенерировать ответ модели без автоматического запуска сервера или обучения."""

    active_config = config or GenerationConfig()
    if not prompt.strip():
        raise ValueError("prompt не может быть пустым")
    device = torch.device(active_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Поддерживаются только устройства cpu и cuda")

    tokenizer = BPETokenizer.load(active_config.tokenizer_file)
    model = _load_model(active_config.checkpoint_file, device)
    prompt_ids = [
        tokenizer.token_to_id(BOS_TOKEN),
        tokenizer.token_to_id(USER_TOKEN),
        *tokenizer.encode(prompt),
        tokenizer.token_to_id(ASSISTANT_TOKEN),
    ]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids,
        max_new_tokens=active_config.max_new_tokens,
        temperature=active_config.temperature,
        top_k=active_config.top_k,
    )
    new_token_ids = cast(list[int], generated[0, len(prompt_ids) :].tolist())
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id in new_token_ids:
        new_token_ids = new_token_ids[: new_token_ids.index(eos_id)]
    return GenerationResult(
        text=tokenizer.decode(new_token_ids, skip_special_tokens=True),
        token_ids=new_token_ids,
    )
