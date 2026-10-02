"""Конфигурация загрузки и адаптации pretrained causal LM."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

AdaptationMode = Literal["full", "lora", "qlora"]
TorchDType = Literal["auto", "float32", "float16", "bfloat16"]


class PretrainedConfig(BaseModel):
    """Воспроизводимые параметры Hugging Face модели, tokenizer и PEFT."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    revision: str = Field(default="main", min_length=1)
    cache_dir: Path = Path(".cache/huggingface")
    adaptation_mode: AdaptationMode = "full"
    torch_dtype: TorchDType = "auto"
    max_sequence_length: int = Field(default=512, ge=2)
    local_files_only: bool = False
    trust_remote_code: bool = False
    gradient_checkpointing: bool = True
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    lora_target_modules: tuple[str, ...] | Literal["all-linear"] = "all-linear"
    qlora_quant_type: Literal["nf4", "fp4"] = "nf4"
    qlora_double_quant: bool = True
    device_map: str | None = None

    @model_validator(mode="before")
    @classmethod
    def apply_qlora_defaults(cls, value: object) -> object:
        """Подставить профиль QLoRA, сохранив явно заданные старые параметры."""

        if isinstance(value, dict) and value.get("adaptation_mode") == "qlora":
            return {"torch_dtype": "bfloat16", "gradient_checkpointing": False, **value}
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        """Проверить параметры, специфичные для режима адаптации."""

        if isinstance(self.lora_target_modules, tuple) and not self.lora_target_modules:
            raise ValueError("lora_target_modules не может быть пустым")
        if self.adaptation_mode == "qlora" and self.torch_dtype == "float32":
            raise ValueError("QLoRA требует auto, float16 или bfloat16 compute dtype")
        return self

    def save(self, path: Path) -> None:
        """Атомарно сохранить параметры в JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> PretrainedConfig:
        """Загрузить и проверить параметры из JSON."""

        try:
            payload = path.read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError(f"Не удалось прочитать pretrained config {path}: {error}") from error
        return cls.model_validate_json(payload)
