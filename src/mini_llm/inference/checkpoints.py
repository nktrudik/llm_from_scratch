"""Безопасный выбор checkpoint по имени внутри каталога активной модели."""

from pathlib import Path


def validate_checkpoint_name(name: str) -> str:
    """Принять только имя .pt файла, без пути и специальных символов Windows."""

    normalized = name.strip()
    if (
        not normalized
        or len(normalized) > 255
        or any(char in normalized for char in '<>:"/\\|?*')
        or any(ord(char) < 32 for char in normalized)
        or Path(normalized).suffix != ".pt"
    ):
        raise ValueError("Укажите только имя .pt файла, например step_00000500.pt, без пути")
    return normalized


def resolve_named_checkpoint(checkpoint_dir: Path, name: str) -> Path:
    """Проверить, что непустой checkpoint действительно находится в нужном каталоге."""

    checkpoint = checkpoint_dir / validate_checkpoint_name(name)
    if checkpoint.resolve().parent != checkpoint_dir.resolve():
        raise ValueError("Checkpoint должен находиться внутри каталога активной модели")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint не найден или пуст: {checkpoint}")
    return checkpoint
