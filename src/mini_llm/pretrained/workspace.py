"""Каталоги модели и регистрация активного pretrained backend проекта."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mini_llm.pretrained.config import AdaptationMode

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ACTIVE_MODEL_FILE = Path("artifacts/pretrained/active.json")
ADAPTATION_MODES: tuple[AdaptationMode, ...] = ("qlora", "lora", "full")


@dataclass(frozen=True, slots=True)
class ModelPaths:
    """Изолировать конфигурацию и checkpoints каждой модели и режима адаптации."""

    config_file: Path
    checkpoint_dir: Path
    registration_file: Path

    @classmethod
    def for_model(cls, model_id: str, mode: AdaptationMode = "qlora") -> ModelPaths:
        """Построить безопасные относительные пути по Hugging Face model ID."""

        parts = model_id.split("/")
        if (
            len(parts) not in {1, 2}
            or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", part) for part in parts)
            or any(".." in part or "--" in part or part.endswith(".") for part in parts)
        ):
            raise ValueError("Ожидался Hugging Face model ID вида owner/model или model")
        # Префикс исключает зарезервированные Windows-имена вроде CON и NUL.
        slug = "model--" + "--".join(parts).lower()
        return cls(
            Path("configs/pretrained") / f"{slug}-{mode}.json",
            Path("checkpoints/pretrained") / slug / mode,
            Path("artifacts/pretrained") / slug / f"{mode}.json",
        )


class ModelRegistration(BaseModel):
    """Ссылки на успешно скачанную revision, её конфиг и каталог SFT."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    revision: str
    config_file: Path
    checkpoint_dir: Path
    snapshot_path: Path

    @property
    def best_checkpoint(self) -> Path:
        """Никогда не подменять best checkpoint произвольным periodic checkpoint."""

        return self.checkpoint_dir / "best.pt"

    def save(self, path: Path) -> None:
        """Атомарно сохранить регистрацию без изменения весов и checkpoint."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def load_active_model() -> ModelRegistration | None:
    """Прочитать регистрацию без загрузки модели, tokenizer или checkpoint."""

    if not ACTIVE_MODEL_FILE.is_file():
        return None
    return _load_registration(ACTIVE_MODEL_FILE)


def _load_registration(path: Path) -> ModelRegistration:
    """Прочитать регистрацию без загрузки весов."""

    try:
        return ModelRegistration.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Не удалось прочитать {path}: {error}") from error


def load_registered_model(model_id: str) -> ModelRegistration | None:
    """Выбрать модель, не изменяя active.json; приоритет режимов: QLoRA, LoRA, full."""

    ModelPaths.for_model(model_id)
    active = load_active_model()
    if active is not None and active.model_id == model_id:
        return active
    for mode in ADAPTATION_MODES:
        path = ModelPaths.for_model(model_id, mode).registration_file
        if path.is_file():
            registration = _load_registration(path)
            if registration.model_id != model_id:
                raise RuntimeError(f"Регистрация {path} не соответствует model ID {model_id}")
            return registration
    return None


def list_registered_models() -> list[ModelRegistration]:
    """Перечислить локальные модели; активная регистрация имеет приоритет."""

    registrations: dict[str, ModelRegistration] = {}
    for mode in ADAPTATION_MODES:
        for path in sorted(ACTIVE_MODEL_FILE.parent.glob(f"model--*/{mode}.json")):
            try:
                registration = _load_registration(path)
                expected = ModelPaths.for_model(registration.model_id, mode).registration_file
                if expected != path:
                    raise ValueError("Путь регистрации не соответствует model ID")
            except (RuntimeError, ValueError) as error:
                logging.getLogger(__name__).warning("Регистрация пропущена: %s", error)
                continue
            registrations.setdefault(registration.model_id, registration)
    active = load_active_model()
    if active is not None:
        registrations[active.model_id] = active
    return [registrations[model_id] for model_id in sorted(registrations)]
