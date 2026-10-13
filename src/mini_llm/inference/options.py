"""Доступность pretrained-режимов без загрузки весов и запуска генерации."""

from dataclasses import dataclass

from mini_llm.pretrained import PretrainedConfig
from mini_llm.pretrained.workspace import DEFAULT_MODEL_ID, load_active_model, load_registered_model


@dataclass(frozen=True, slots=True)
class GenerationAvailability:
    """Состояние выбранной pretrained-модели для минимального UI."""

    model_id: str
    before_sft_available: bool
    after_sft_available: bool
    reason: str | None


def pretrained_generation_options(model_id: str | None = None) -> GenerationAvailability:
    """Проверить выбранную либо активную модель без загрузки весов."""

    try:
        registration = load_registered_model(model_id) if model_id else load_active_model()
        if registration is None:
            return GenerationAvailability(
                model_id or DEFAULT_MODEL_ID,
                False,
                False,
                "Сначала выполните pretrained setup <model-id>.",
            )
        config = PretrainedConfig.load(registration.config_file)
        if config.model_id != registration.model_id or config.revision != registration.revision:
            raise RuntimeError("Конфиг не совпадает с регистрацией модели; повторите setup.")
        if not registration.snapshot_path.is_dir():
            raise RuntimeError("Локальные файлы модели не найдены; повторите setup.")
        after_sft = (
            registration.best_checkpoint.is_file()
            and registration.best_checkpoint.stat().st_size > 0
        )
        return GenerationAvailability(
            registration.model_id,
            True,
            after_sft,
            None if after_sft else "После SFT недоступен: best.pt ещё не создан.",
        )
    except (OSError, RuntimeError, ValueError) as error:
        return GenerationAvailability(model_id or DEFAULT_MODEL_ID, False, False, str(error))
