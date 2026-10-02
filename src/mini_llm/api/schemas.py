"""Pydantic-схемы входов и ответов локального HTTP API."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mini_llm.data.config import MAX_BATCH_SIZE
from mini_llm.data.preprocessing import PreprocessingConfig
from mini_llm.data.scraping import ScraperConfig
from mini_llm.inference import GenerationConfig
from mini_llm.inference.checkpoints import resolve_named_checkpoint, validate_checkpoint_name
from mini_llm.inference.config import (
    DEFAULT_CUSTOM_CHECKPOINT_PATH,
    DEFAULT_PRETRAINED_CHECKPOINT_PATH,
    DEFAULT_PRETRAINED_CONFIG_PATH,
    PretrainedGenerationMode,
)
from mini_llm.modeling.config import DEFAULT_MAX_SEQUENCE_LENGTH
from mini_llm.pretrained import PretrainedConfig
from mini_llm.pretrained.workspace import load_active_model
from mini_llm.tokenization.config import DEFAULT_TOKENIZER_PATH
from mini_llm.training import PretrainedTrainingConfig, TrainingConfig


class HealthResponse(BaseModel):
    """Состояние процесса API и текущей тяжёлой задачи."""

    status: str
    active_job_id: str | None


class JobResponse(BaseModel):
    """Снимок состояния фоновой задачи."""

    job_id: str
    kind: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: object | None = None
    error: str | None = None


class TrainingRequest(BaseModel):
    """Параметры запуска полного обучения."""

    splits_dir: Path = Field(default_factory=lambda: Path("data/processed/splits"))
    tokenizer_file: Path = Field(default_factory=lambda: DEFAULT_TOKENIZER_PATH)
    token_statistics_file: Path = Field(
        default_factory=lambda: Path("data/processed/token_statistics.json")
    )
    checkpoint_dir: Path = Field(default_factory=lambda: Path("checkpoints/training"))
    resume_from: Path | None = None
    model_backend: str = Field(default="custom", pattern="^(custom|pretrained)$")
    pretrained_config_file: Path | None = None
    batch_size: int = Field(default=MAX_BATCH_SIZE, ge=1, le=MAX_BATCH_SIZE)
    num_workers: int = Field(default=0, ge=0)
    epochs: int | None = Field(default=3, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    validation_interval: int = Field(default=1000, ge=1)
    checkpoint_interval: int = Field(default=1000, ge=1)
    validation_batches: int = Field(default=200, ge=0)
    log_interval: int = Field(default=10, ge=1)
    random_seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    max_train_samples: int | None = Field(default=None, ge=1)
    max_validation_samples: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_duration(self) -> Self:
        """Не допустить одновременный лимит по эпохам и шагам."""

        if (self.epochs is None) == (self.max_steps is None):
            raise ValueError("Нужно задать ровно один режим: epochs или max_steps")
        if self.model_backend == "pretrained" and self.pretrained_config_file is None:
            raise ValueError("Для pretrained backend нужен pretrained_config_file")
        if self.model_backend == "custom" and self.pretrained_config_file is not None:
            raise ValueError("pretrained_config_file допустим только для pretrained backend")
        if self.model_backend == "pretrained":
            defaults = PretrainedTrainingConfig(pretrained_config_file=self.pretrained_config_file)
            for name in (
                "batch_size",
                "num_workers",
                "max_train_samples",
                "max_validation_samples",
                "log_interval",
                "validation_interval",
                "validation_batches",
                "checkpoint_interval",
            ):
                if name not in self.model_fields_set:
                    setattr(self, name, getattr(defaults, name))
        return self

    def to_config(self) -> TrainingConfig:
        """Преобразовать HTTP-схему во внутреннюю конфигурацию training pipeline."""

        config_class = (
            PretrainedTrainingConfig if self.model_backend == "pretrained" else TrainingConfig
        )
        return config_class(**self.model_dump())


class PretrainedPrepareRequest(BaseModel):
    """Параметры подготовки модели из Hugging Face Hub или локального cache."""

    model_id: str = Field(min_length=1)
    revision: str = Field(default="main", min_length=1)
    cache_dir: Path = Field(default_factory=lambda: Path(".cache/huggingface"))
    adaptation_mode: str = Field(default="lora", pattern="^(full|lora|qlora)$")
    torch_dtype: str = Field(default="auto", pattern="^(auto|float32|float16|bfloat16)$")
    max_sequence_length: int = Field(default=512, ge=2)
    local_files_only: bool = False
    trust_remote_code: bool = False
    gradient_checkpointing: bool = True
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_target_modules: tuple[str, ...] | str = "all-linear"
    qlora_quant_type: str = Field(default="nf4", pattern="^(nf4|fp4)$")
    qlora_double_quant: bool = True
    device_map: str | None = None
    output_config: Path = Field(default_factory=lambda: Path("configs/pretrained/model.json"))

    @model_validator(mode="after")
    def apply_qlora_defaults(self) -> Self:
        """Согласовать HTTP-defaults QLoRA с конфигом модели, не меняя явные значения."""

        if self.adaptation_mode == "qlora":
            if "torch_dtype" not in self.model_fields_set:
                self.torch_dtype = "bfloat16"
            if "gradient_checkpointing" not in self.model_fields_set:
                self.gradient_checkpointing = False
        return self

    def to_config(self) -> PretrainedConfig:
        """Преобразовать HTTP-схему в проверенную pretrained-конфигурацию."""

        return PretrainedConfig.model_validate(self.model_dump(exclude={"output_config"}))


class TokenizerTrainingRequest(BaseModel):
    """Параметры обучения BPE tokenizer."""

    train_file: Path = Field(default_factory=lambda: Path("data/processed/splits/train.jsonl"))
    output_file: Path = Field(default_factory=lambda: DEFAULT_TOKENIZER_PATH)
    vocab_size: int = Field(default=8192, ge=262)
    min_frequency: int = Field(default=2, ge=1)


class PreprocessingRequest(BaseModel):
    """Параметры preprocessing raw-тредов."""

    input_dir: Path = Field(default_factory=lambda: Path("data/raw/2ch"))
    output_dir: Path = Field(default_factory=lambda: Path("data/processed"))
    review_size: int = Field(default=100, ge=0)
    random_seed: int = 42
    dedup_min_characters: int = Field(default=80, ge=1)
    near_duplicate_threshold: float = Field(default=0.9, gt=0, le=1)

    def to_config(self) -> PreprocessingConfig:
        """Преобразовать запрос во внутреннюю конфигурацию preprocessing."""

        return PreprocessingConfig(**self.model_dump())


class DatasetSplitRequest(BaseModel):
    """Параметры детерминированного split по тредам."""

    input_file: Path = Field(default_factory=lambda: Path("data/processed/2ch_dialogues.jsonl"))
    output_dir: Path = Field(default_factory=lambda: Path("data/processed/splits"))
    random_seed: int = 42


class TokenStatisticsRequest(BaseModel):
    """Параметры подсчёта token statistics."""

    splits_dir: Path = Field(default_factory=lambda: Path("data/processed/splits"))
    tokenizer_file: Path = Field(default_factory=lambda: DEFAULT_TOKENIZER_PATH)
    output_file: Path = Field(default_factory=lambda: Path("data/processed/token_statistics.json"))
    max_sequence_length: int = Field(default=DEFAULT_MAX_SEQUENCE_LENGTH, ge=1)


class ScrapeRequest(BaseModel):
    """Параметры последовательного сбора тредов 2ch."""

    url: str
    output_dir: Path = Field(default_factory=lambda: Path("data/raw"))
    max_threads: int | None = Field(default=None, ge=1)
    timeout_seconds: float = Field(default=15.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    backoff_factor: float = Field(default=1.0, ge=0)
    min_request_delay: float = Field(default=1.0, ge=0)
    max_request_delay: float = Field(default=2.5, ge=0)

    def to_config(self) -> ScraperConfig:
        """Преобразовать запрос в сетевую конфигурацию scraper."""

        values = self.model_dump(exclude={"url", "max_threads"})
        return ScraperConfig(**values)


class GenerationRequest(BaseModel):
    """Prompt и параметры генерации ответа."""

    prompt: str = Field(min_length=1)
    model_backend: Literal["custom", "pretrained"] = "custom"
    checkpoint_file: Path | None = Field(default_factory=lambda: DEFAULT_CUSTOM_CHECKPOINT_PATH)
    tokenizer_file: Path | None = Field(default_factory=lambda: DEFAULT_TOKENIZER_PATH)
    pretrained_config_file: Path | None = None
    pretrained_mode: PretrainedGenerationMode | None = None
    checkpoint_name: str | None = Field(
        default=None,
        description="Имя .pt файла в каталоге активной модели для custom_checkpoint",
    )
    device: str = "cuda"
    max_new_tokens: int = Field(default=256, ge=0)
    temperature: float = Field(default=0.3, gt=0)
    top_k: int | None = Field(default=20, ge=1)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_backend": "custom",
                    "prompt": "Привет! Как у тебя дела?",
                    "checkpoint_file": str(DEFAULT_CUSTOM_CHECKPOINT_PATH),
                    "tokenizer_file": str(DEFAULT_TOKENIZER_PATH),
                },
                {
                    "model_backend": "pretrained",
                    "prompt": "Привет! Как у тебя дела?",
                    "pretrained_mode": "after_sft",
                    "checkpoint_file": str(DEFAULT_PRETRAINED_CHECKPOINT_PATH),
                    "pretrained_config_file": str(DEFAULT_PRETRAINED_CONFIG_PATH),
                    "device": "cuda",
                    "max_new_tokens": 256,
                    "temperature": 0.3,
                    "top_k": 20,
                },
                {
                    "model_backend": "pretrained",
                    "prompt": "Привет! Как у тебя дела?",
                    "pretrained_mode": "custom_checkpoint",
                    "checkpoint_name": "step_00000500.pt",
                },
            ]
        }
    )

    @model_validator(mode="after")
    def validate_backend_files(self) -> Self:
        """Подставить defaults выбранного backend и проверить обязательные файлы."""

        if self.checkpoint_name is not None:
            if self.model_backend != "pretrained" or self.pretrained_mode != "custom_checkpoint":
                raise ValueError("checkpoint_name допустим только в режиме custom_checkpoint Qwen")
            self.checkpoint_name = validate_checkpoint_name(self.checkpoint_name)
        if self.pretrained_mode == "custom_checkpoint":
            if self.model_backend != "pretrained" or self.checkpoint_name is None:
                raise ValueError("Для custom_checkpoint Qwen нужен checkpoint_name")
            if "checkpoint_file" in self.model_fields_set:
                raise ValueError(
                    "В custom_checkpoint передавайте checkpoint_name, не checkpoint_file"
                )
            if "pretrained_config_file" in self.model_fields_set:
                raise ValueError("В custom_checkpoint конфиг определяется активной моделью")
        if self.model_backend == "pretrained":
            # Явно переданные пути сохраняются; defaults выбирает API, а не UI.
            explicit_checkpoint = "checkpoint_file" in self.model_fields_set
            if self.pretrained_mode is None:
                self.pretrained_mode = (
                    "after_sft"
                    if explicit_checkpoint and self.checkpoint_file is not None
                    else "before_sft"
                )
            registration = (
                load_active_model()
                if not explicit_checkpoint or "pretrained_config_file" not in self.model_fields_set
                else None
            )
            if not explicit_checkpoint:
                self.checkpoint_file = (
                    (
                        registration.best_checkpoint
                        if registration
                        else DEFAULT_PRETRAINED_CHECKPOINT_PATH
                    )
                    if self.pretrained_mode == "after_sft"
                    else None
                )
                if self.pretrained_mode == "custom_checkpoint" and self.checkpoint_name is not None:
                    checkpoint_dir = (
                        registration.checkpoint_dir
                        if registration
                        else DEFAULT_PRETRAINED_CHECKPOINT_PATH.parent
                    )
                    self.checkpoint_file = checkpoint_dir / self.checkpoint_name
            if "pretrained_config_file" not in self.model_fields_set:
                self.pretrained_config_file = (
                    registration.config_file if registration else DEFAULT_PRETRAINED_CONFIG_PATH
                )
            if "tokenizer_file" not in self.model_fields_set:
                self.tokenizer_file = None
        if self.model_backend == "custom" and self.tokenizer_file is None:
            raise ValueError("Для custom backend нужен tokenizer_file")
        if self.model_backend == "pretrained" and self.pretrained_config_file is None:
            raise ValueError("Для pretrained backend нужен pretrained_config_file")
        if self.model_backend == "custom" and self.checkpoint_file is None:
            raise ValueError("Для custom backend нужен checkpoint_file")
        if self.model_backend == "custom" and self.pretrained_mode is not None:
            raise ValueError("pretrained_mode допустим только для pretrained backend")
        if self.model_backend == "pretrained":
            if (
                self.pretrained_mode in {"after_sft", "custom_checkpoint"}
                and self.checkpoint_file is None
            ):
                raise ValueError(f"Для режима {self.pretrained_mode} нужен checkpoint_file")
            if self.pretrained_mode == "before_sft" and self.checkpoint_file is not None:
                raise ValueError("Режим before_sft не использует checkpoint_file")
        return self

    def to_config(self) -> GenerationConfig:
        """Преобразовать параметры запроса во внутреннюю конфигурацию генерации."""

        values = self.model_dump(exclude={"prompt", "checkpoint_name"})
        if self.pretrained_mode == "custom_checkpoint":
            if self.checkpoint_file is None or self.checkpoint_name is None:
                raise ValueError("Для custom_checkpoint нужен checkpoint_name")
            values["checkpoint_file"] = resolve_named_checkpoint(
                self.checkpoint_file.parent, self.checkpoint_name
            )
        if values["pretrained_mode"] is None:
            values.pop("pretrained_mode")
        return GenerationConfig(**values)


class GenerationResponse(BaseModel):
    """Сгенерированный текст и новые token IDs."""

    text: str
    token_ids: list[int]


class GenerationOptionsResponse(BaseModel):
    """Доступные pretrained-режимы; проверка не загружает модель или checkpoint."""

    model_id: str
    before_sft_available: bool
    after_sft_available: bool
    reason: str | None
