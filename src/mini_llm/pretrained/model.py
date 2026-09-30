"""Загрузка Hugging Face causal LM и подготовка full/LoRA/QLoRA."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from mini_llm.modeling import CausalLMBackend, CausalLMOutput
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.dependencies import require_module
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer


class _ResizableModel(Protocol):
    def resize_token_embeddings(self, new_num_tokens: int) -> object: ...

    def gradient_checkpointing_enable(self) -> None: ...


class _ModelConfig(Protocol):
    use_cache: bool
    pad_token_id: int


class _FromPretrained(Protocol):
    def from_pretrained(self, model_id: str, **kwargs: object) -> object: ...


class HuggingFaceCausalLMBackend(CausalLMBackend):
    """Унифицированный backend для Transformers и PEFT моделей."""

    def __init__(self, model: nn.Module, config: PretrainedConfig) -> None:
        self._model = model
        self.config = config

    @property
    def module(self) -> nn.Module:
        return self._model

    @property
    def max_sequence_length(self) -> int:
        return self.config.max_sequence_length

    @property
    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "backend": "pretrained",
            "pretrained_config": self.config.model_dump(mode="json"),
        }

    @property
    def autocast_dtype(self) -> torch.dtype:
        """Согласовать AMP с compute dtype pretrained-конфигурации."""

        return torch.bfloat16 if self.config.torch_dtype == "bfloat16" else torch.float16

    def to(self, device: torch.device) -> None:
        """Не перемещать повторно 4-bit модель, уже размещённую Accelerate."""

        if self.config.adaptation_mode != "qlora":
            super().to(device)

    def forward_batch(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> CausalLMOutput:
        call_model = cast(Callable[..., object], self._model)
        output = call_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        logits = getattr(output, "logits", None)
        loss = getattr(output, "loss", None)
        if not isinstance(logits, Tensor) or not isinstance(loss, Tensor):
            raise RuntimeError("Hugging Face causal LM не вернула logits и loss")
        return CausalLMOutput(logits, loss)

    def checkpoint_state_dict(self) -> Mapping[str, Tensor]:
        """Для PEFT сохранять только компактные параметры адаптера."""

        if self.config.adaptation_mode == "full":
            return super().checkpoint_state_dict()
        peft = require_module("peft")
        getter = cast(Callable[[nn.Module], object], peft.get_peft_model_state_dict)
        state = getter(self._model)
        if not isinstance(state, dict):
            raise RuntimeError("PEFT вернул некорректный state dict")
        return cast(dict[str, Tensor], state)

    def load_checkpoint_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Загрузить полный state или параметры PEFT-адаптера."""

        if self.config.adaptation_mode == "full":
            super().load_checkpoint_state_dict(state)
            return
        peft = require_module("peft")
        setter = cast(
            Callable[[nn.Module, Mapping[str, Tensor]], object],
            peft.set_peft_model_state_dict,
        )
        setter(self._model, state)


@dataclass(frozen=True, slots=True)
class PreparedPretrained:
    """Совместимые model backend и tokenizer после подготовки."""

    model: HuggingFaceCausalLMBackend
    tokenizer: HuggingFaceDialogueTokenizer

    @property
    def trainable_parameters(self) -> int:
        """Вернуть число параметров, обновляемых при fine-tuning."""

        return sum(parameter.numel() for parameter in self.model.trainable_parameters())

    @property
    def total_parameters(self) -> int:
        """Вернуть полное число параметров модели."""

        return sum(parameter.numel() for parameter in self.model.module.parameters())


def _from_pretrained(factory: object, model_id: str, **kwargs: object) -> object:
    return cast(_FromPretrained, factory).from_pretrained(model_id, **kwargs)


def _torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def prepare_pretrained_model(config: PretrainedConfig) -> PreparedPretrained:
    """Загрузить модель/tokenizer и применить выбранный режим fine-tuning."""

    transformers = require_module("transformers")
    tokenizer_factory = transformers.AutoTokenizer
    tokenizer_object = _from_pretrained(
        tokenizer_factory,
        config.model_id,
        revision=config.revision,
        cache_dir=str(config.cache_dir),
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    tokenizer = HuggingFaceDialogueTokenizer(tokenizer_object)

    model_kwargs: dict[str, object] = {
        "revision": config.revision,
        "cache_dir": str(config.cache_dir),
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
        "dtype": _torch_dtype(config.torch_dtype),
    }
    if config.adaptation_mode == "qlora":
        quantization_factory = cast(Callable[..., object], transformers.BitsAndBytesConfig)
        model_kwargs["quantization_config"] = quantization_factory(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.qlora_quant_type,
            bnb_4bit_use_double_quant=config.qlora_double_quant,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float16
            ),
        )
        model_kwargs["device_map"] = config.device_map or "auto"

    model_factory = transformers.AutoModelForCausalLM
    model_object = _from_pretrained(model_factory, config.model_id, **model_kwargs)
    if not isinstance(model_object, nn.Module):
        raise RuntimeError("AutoModelForCausalLM вернул объект, не являющийся nn.Module")
    model = model_object
    resizable_model = cast(_ResizableModel, model)
    resizable_model.resize_token_embeddings(tokenizer.vocab_size)

    model_config = cast(_ModelConfig | None, getattr(model, "config", None))
    if model_config is not None:
        model_config.use_cache = False
        model_config.pad_token_id = tokenizer.pad_token_id
    if config.gradient_checkpointing:
        resizable_model.gradient_checkpointing_enable()

    if config.adaptation_mode in {"lora", "qlora"}:
        peft = require_module("peft")
        if config.adaptation_mode == "qlora":
            prepare_kbit = cast(Callable[..., nn.Module], peft.prepare_model_for_kbit_training)
            model = prepare_kbit(
                model,
                use_gradient_checkpointing=config.gradient_checkpointing,
            )
        lora_factory = cast(Callable[..., object], peft.LoraConfig)
        lora_config = lora_factory(
            task_type="CAUSAL_LM",
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=(
                config.lora_target_modules
                if isinstance(config.lora_target_modules, str)
                else list(config.lora_target_modules)
            ),
            bias="none",
        )
        apply_peft = cast(Callable[[nn.Module, object], nn.Module], peft.get_peft_model)
        model = apply_peft(model, lora_config)

    return PreparedPretrained(HuggingFaceCausalLMBackend(model, config), tokenizer)


def save_pretrained_parameters(config: PretrainedConfig, path: Path) -> None:
    """Сохранить фактически использованную конфигурацию pretrained-модели."""

    config.save(path)
