"""Актуальные модели и запуски с оценкой места на диске, без загрузки весов."""

from pathlib import Path

from mini_llm.inference.checkpoints import resolve_named_checkpoint
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.management.schemas import ManagedModel, ManagedRun, ModelsCatalog
from mini_llm.pretrained.management.storage import CACHE_ROOT, inspect_path, model_paths
from mini_llm.pretrained.workspace import ModelPaths, list_registered_models, load_active_model
from mini_llm.workspace.catalog import list_runs


def models_catalog(active_job_id: str | None = None) -> ModelsCatalog:
    """Читать локальный реестр; запрос не обращается к Hugging Face или GPU."""

    registrations = {item.model_id: item for item in list_registered_models()}
    runs = list_runs()
    ids = {item.model_id.casefold(): item.model_id for item in runs}
    warnings: list[str] = []
    for path in Path("configs/models").glob("model--*/*.json"):
        try:
            config = PretrainedConfig.load(path)
            if path != ModelPaths.for_model(config.model_id, config.adaptation_mode).config_file:
                raise ValueError(f"Конфиг находится в неправильном каталоге: {path}")
            ids.setdefault(config.model_id.casefold(), config.model_id)
        except (ValueError, RuntimeError, OSError) as error:
            warnings.append(str(error))
    for cache in CACHE_ROOT.glob("models--*"):
        model_id = cache.name.removeprefix("models--").replace("--", "/")
        try:
            ModelPaths.for_model(model_id)
            ids.setdefault(model_id.casefold(), model_id)
        except ValueError:
            warnings.append(f"Неизвестный каталог HF cache: {cache}")
    ids.update({model_id.casefold(): model_id for model_id in registrations})
    ids["custom"] = "custom"
    active = load_active_model()
    models: list[ManagedModel] = []
    for model_id in sorted(ids.values()):
        registration = registrations.get(model_id)
        model = ManagedModel(
            model_id=model_id,
            registered=registration is not None,
            available=model_id == "custom"
            or bool(registration and registration.snapshot_path.is_dir()),
            active=bool(active and active.model_id == model_id),
            revision=registration.revision if registration else None,
            config_file=registration.config_file.as_posix() if registration else None,
            snapshot_path=registration.snapshot_path.as_posix() if registration else None,
        )
        try:
            paths = model_paths(model_id, include_cache=model_id != "custom")
        except (OSError, ValueError, RuntimeError) as error:
            warnings.append(f"{model_id}: {error}; внешний cache не включён в размер")
            paths = model_paths(model_id, include_cache=False)
        for path, root in paths:
            if path.exists():
                try:
                    model.size_bytes += inspect_path(path, root)[0].size_bytes
                except (OSError, ValueError) as error:
                    warnings.append(str(error))
        for run in runs:
            if run.model_id.casefold() != model_id.casefold():
                continue
            checkpoints = []
            for path in sorted(run.directory.glob("*.pt")):
                try:
                    resolve_named_checkpoint(run.directory, path.name)
                    checkpoints.append(path.name)
                except (OSError, ValueError):
                    continue
            try:
                size = inspect_path(run.directory, run.directory.parent)[0].size_bytes
            except (OSError, ValueError):
                size = 0
            model.runs.append(
                ManagedRun(
                    run_id=run.run_id,
                    title=run.title,
                    model_id=model_id,
                    directory=run.directory.as_posix(),
                    checkpoints=checkpoints,
                    size_bytes=size,
                )
            )
        models.append(model)
    return ModelsCatalog(
        models=models,
        active_job_id=active_job_id,
        total_size_bytes=sum(item.size_bytes for item in models),
        warnings=warnings,
    )
