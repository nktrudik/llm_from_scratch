"""Одна модель в памяти и последовательная замена полных весов либо PEFT-адаптера."""

import gc
from dataclasses import asdict
from pathlib import Path

import torch

from mini_llm.inference.config import GenerationConfig
from mini_llm.inference.schemas import GenerationResult
from mini_llm.inference.service import (
    _checkpoint_model_state,
    _generate_custom_response,
    _load_custom_model,
    _read_checkpoint,
)
from mini_llm.modeling import DecoderOnlyTransformer, ModelConfig
from mini_llm.pretrained import PreparedPretrained, PretrainedConfig, prepare_pretrained_model
from mini_llm.pretrained.inference import generate_pretrained
from mini_llm.tokenization import BPETokenizer
from mini_llm.workspace.catalog import ModelRun


class CheckpointSession:
    """Не загружать base model и tokenizer повторно для каждого вопроса или адаптера."""

    def __init__(self, run: ModelRun, config: GenerationConfig) -> None:
        self.run = run
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна; автоматического перехода на CPU нет")
        self.prepared: PreparedPretrained | None = None
        self.custom: DecoderOnlyTransformer | None = None
        self.tokenizer: BPETokenizer | None = None

    def load(self, checkpoint: Path) -> None:
        """Проверить совместимость и полноту state перед заменой весов текущей модели."""

        if self.run.model_id == "custom":
            if self.custom is None:
                if self.run.tokenizer_file is None:
                    raise ValueError("У запуска Custom отсутствует tokenizer_file")
                self.tokenizer = BPETokenizer.load(self.run.tokenizer_file)
                self.custom = _load_custom_model(checkpoint, self.device)
            else:
                payload = _read_checkpoint(checkpoint)
                if payload.get("model_config") != asdict(ModelConfig()):
                    raise ValueError("ModelConfig checkpoint не совпадает с текущей архитектурой")
                self.custom.load_state_dict(_checkpoint_model_state(payload))
                self.custom.eval()
            return
        if self.run.pretrained_config_file is None:
            raise ValueError("У запуска нет pretrained_config_file")
        base = PretrainedConfig.load(self.run.pretrained_config_file)
        payload = _read_checkpoint(checkpoint)
        metadata = {"backend": "pretrained", "pretrained_config": base.model_dump(mode="json")}
        if payload.get("model_metadata") != metadata:
            raise ValueError(f"Checkpoint {checkpoint.name} несовместим с конфигом запуска")
        if self.prepared is None:
            self.prepared = prepare_pretrained_model(base, for_inference=True)
            self.prepared.model.to(self.device)
        if self.prepared.model.checkpoint_metadata != metadata:
            raise ValueError("Конфиг запуска изменился во время проверки")
        state = _checkpoint_model_state(payload)
        expected = self.prepared.model.checkpoint_state_dict()
        if state.keys() != expected.keys() or any(
            state[key].shape != expected[key].shape for key in state
        ):
            raise ValueError("Checkpoint содержит неполный или несовместимый model_state_dict")
        self.prepared.model.load_checkpoint_state_dict(state)
        self.prepared.model.eval()

    def generate(self, prompt: str) -> GenerationResult:
        """Использовать существующую генерацию, system prompt и non-thinking template."""

        if self.prepared is not None:
            result = generate_pretrained(
                self.prepared,
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
            )
            return GenerationResult(result.text, result.token_ids)
        if self.custom is None or self.tokenizer is None:
            raise RuntimeError("Сначала загрузите checkpoint")
        return _generate_custom_response(
            prompt, self.config, self.device, self.custom, self.tokenizer
        )

    def close(self) -> None:
        """Освободить GPU и CPU даже при остановке или ошибке в середине прогона."""

        self.prepared = None
        self.custom = None
        self.tokenizer = None
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
