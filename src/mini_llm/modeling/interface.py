"""Общий интерфейс causal language model для training и inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import Tensor, nn

from mini_llm.modeling.config import ModelConfig


@dataclass(frozen=True, slots=True)
class CausalLMOutput:
    """Унифицированный результат forward-pass языковой модели."""

    logits: Tensor
    loss: Tensor


class CausalLMBackend(ABC):
    """Адаптер, скрывающий детали конкретной реализации causal LM."""

    @property
    @abstractmethod
    def module(self) -> nn.Module:
        """Вернуть базовый PyTorch-модуль."""

    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """Вернуть максимальную длину обучающего окна."""

    @property
    @abstractmethod
    def checkpoint_metadata(self) -> dict[str, object]:
        """Вернуть параметры, необходимые для проверки checkpoint."""

    @property
    def autocast_dtype(self) -> torch.dtype:
        """Вернуть dtype AMP для текущего backend."""

        return torch.float16

    @abstractmethod
    def forward_batch(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> CausalLMOutput:
        """Выполнить forward-pass и вернуть logits вместе с loss."""

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Вернуть только параметры, обновляемые optimizer."""

        return (parameter for parameter in self.module.parameters() if parameter.requires_grad)

    def to(self, device: torch.device) -> None:
        """Переместить модель на устройство."""

        self.module.to(device)

    def train(self) -> None:
        """Перевести модель в режим обучения."""

        self.module.train()

    def eval(self) -> None:
        """Перевести модель в режим оценки."""

        self.module.eval()

    def checkpoint_state_dict(self) -> Mapping[str, Tensor]:
        """Вернуть состояние модели для checkpoint."""

        return self.module.state_dict()

    def load_checkpoint_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Восстановить состояние модели из checkpoint."""

        self.module.load_state_dict(state)


class CustomCausalLMBackend(CausalLMBackend):
    """Адаптер собственной реализации decoder-only Transformer."""

    def __init__(self, model: nn.Module, config: ModelConfig) -> None:
        self._model = model
        self._config = config

    @property
    def module(self) -> nn.Module:
        return self._model

    @property
    def max_sequence_length(self) -> int:
        return self._config.max_sequence_length

    @property
    def checkpoint_metadata(self) -> dict[str, object]:
        return {"backend": "custom", "model_config": asdict(self._config)}

    def forward_batch(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> CausalLMOutput:
        del attention_mask
        call_model = cast(Callable[[Tensor, Tensor], object], self._model)
        result = call_model(input_ids, labels)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("Custom causal LM должна вернуть пару logits/loss")
        logits, loss = result
        if not isinstance(logits, Tensor) or not isinstance(loss, Tensor):
            raise RuntimeError("Custom causal LM вернула некорректный результат")
        return CausalLMOutput(logits, loss)
