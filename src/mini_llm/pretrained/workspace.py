"""Каталоги модели и регистрация активного pretrained backend проекта."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mini_llm.pretrained.config import AdaptationMode

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ACTIVE_MODEL_FILE = Path("artifacts/pretrained/active.json")


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
    try:
        return ModelRegistration.model_validate_json(ACTIVE_MODEL_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Не удалось прочитать {ACTIVE_MODEL_FILE}: {error}") from error
