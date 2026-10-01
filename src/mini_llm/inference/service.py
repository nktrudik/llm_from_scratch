"""Загрузка checkpoint и генерация текстового ответа по одному prompt."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from mini_llm.data.dialogue import ASSISTANT_TOKEN, BOS_TOKEN, EOS_TOKEN, USER_TOKEN
from mini_llm.inference.cache import PretrainedModelCache
from mini_llm.inference.config import GenerationConfig
from mini_llm.inference.schemas import GenerationResult
from mini_llm.modeling import DecoderOnlyTransformer, ModelConfig
from mini_llm.pretrained import (
    PreparedPretrained,
    PretrainedConfig,
    generate_pretrained,
    prepare_pretrained_model,
)
from mini_llm.tokenization import BPETokenizer

_pretrained_cache = PretrainedModelCache()


def clear_pretrained_cache() -> None:
    """Освободить inference-модель, чтобы она не занимала VRAM во время обучения."""

    _pretrained_cache.clear()


def _read_checkpoint(checkpoint_file: Path) -> dict[str, object]:
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
    if not all(isinstance(key, str) for key in payload_object):
        raise RuntimeError("Ключи checkpoint должны быть строками")
    return cast(dict[str, object], payload_object)


def _checkpoint_model_state(payload: dict[str, object]) -> Mapping[str, Tensor]:
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Checkpoint не содержит model_state_dict")
    if not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()):
        raise RuntimeError("model_state_dict checkpoint содержит некорректные значения")
    return cast(Mapping[str, Tensor], state)


def _load_custom_model(checkpoint_file: Path, device: torch.device) -> DecoderOnlyTransformer:
    payload = _read_checkpoint(checkpoint_file)
    saved_config = payload.get("model_config")
    model_config = ModelConfig()
    if not isinstance(saved_config, dict) or saved_config != asdict(model_config):
        raise RuntimeError("ModelConfig checkpoint не совпадает с текущей архитектурой")
    model = DecoderOnlyTransformer(model_config)
    model.load_state_dict(_checkpoint_model_state(payload))
    model.to(device)
    model.eval()
    return model


def _generate_custom(
    prompt: str,
    config: GenerationConfig,
    device: torch.device,
) -> GenerationResult:
    tokenizer_file = config.tokenizer_file
    if tokenizer_file is None:
        raise RuntimeError("Для custom backend не указан tokenizer_file")
    if config.checkpoint_file is None:
        raise RuntimeError("Для custom backend не указан checkpoint_file")
    tokenizer = BPETokenizer.load(tokenizer_file)
    model = _load_custom_model(config.checkpoint_file, device)
    prompt_ids = [
        tokenizer.token_to_id(BOS_TOKEN),
        tokenizer.token_to_id(USER_TOKEN),
        *tokenizer.encode(prompt),
        tokenizer.token_to_id(ASSISTANT_TOKEN),
    ]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
    )
    new_token_ids = cast(list[int], generated[0, len(prompt_ids) :].tolist())
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id in new_token_ids:
        new_token_ids = new_token_ids[: new_token_ids.index(eos_id)]
    return GenerationResult(
        text=tokenizer.decode(new_token_ids, skip_special_tokens=True),
        token_ids=new_token_ids,
    )


def _load_pretrained_model(
    config_file: Path,
    checkpoint_file: Path | None,
    device: torch.device,
) -> PreparedPretrained:
    pretrained_config = PretrainedConfig.load(config_file)
    if checkpoint_file is None:
        # До SFT используется исходная модель без случайно созданного LoRA-адаптера.
        pretrained_config = pretrained_config.model_copy(update={"adaptation_mode": "full"})
    elif not checkpoint_file.is_file():
        raise RuntimeError("Режим после SFT недоступен: checkpoint ещё не создан")
    prepared = prepare_pretrained_model(pretrained_config, for_inference=True)
    if checkpoint_file is not None:
        payload = _read_checkpoint(checkpoint_file)
        saved_metadata = payload.get("model_metadata")
        if saved_metadata != prepared.model.checkpoint_metadata:
            raise RuntimeError("PretrainedConfig не совпадает с model metadata checkpoint")
        prepared.model.load_checkpoint_state_dict(_checkpoint_model_state(payload))
    prepared.model.to(device)
    prepared.model.eval()
    return prepared


def _generate_with_pretrained(
    prompt: str,
    config: GenerationConfig,
    device: torch.device,
) -> GenerationResult:
    config_file = config.pretrained_config_file
    if config_file is None:
        raise RuntimeError("Для pretrained backend не указан pretrained_config_file")
    if config.checkpoint_file is not None and not config.checkpoint_file.is_file():
        raise RuntimeError("Режим после SFT недоступен: checkpoint ещё не создан")
    with _pretrained_cache.use(
        config_file,
        config.checkpoint_file,
        device,
        lambda: _load_pretrained_model(config_file, config.checkpoint_file, device),
    ) as prepared:
        result = generate_pretrained(
            prepared,
            prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
        )
    return GenerationResult(result.text, result.token_ids)


def generate_response(prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
    """Сгенерировать ответ через выбранный custom или pretrained backend."""

    active_config = config or GenerationConfig()
    if not prompt.strip():
        raise ValueError("prompt не может быть пустым")
    device = torch.device(active_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Поддерживаются только устройства cpu и cuda")
    if active_config.model_backend == "pretrained":
        return _generate_with_pretrained(prompt, active_config, device)
    clear_pretrained_cache()
    return _generate_custom(prompt, active_config, device)
