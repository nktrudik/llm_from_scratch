"""Версионируемое обучение поверх существующего trainer, без изменения его алгоритма."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mini_llm.data.config import MAX_BATCH_SIZE
from mini_llm.data.dialogue import iter_dialogue_jsonl
from mini_llm.data.splitting import SplitConfig, split_dataset
from mini_llm.modeling.config import ModelConfig
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.workspace import ModelPaths, load_registered_model
from mini_llm.tokenization.config import DEFAULT_TOKENIZER_PATH
from mini_llm.training.config import PretrainedTrainingConfig, TrainingConfig
from mini_llm.training.pipeline import train_model
from mini_llm.training.schemas import TrainingResult
from mini_llm.workspace.catalog import (
    DatasetVersion,
    ModelRun,
    fingerprint,
    get_dataset,
    get_run,
    list_datasets,
    list_runs,
    reserve_run,
)


class TrainingParameters(BaseModel):
    """Только изменяемые параметры обучения, без произвольных путей клиента."""

    model_config = ConfigDict(extra="forbid")
    batch_size: int = Field(default=2, ge=1, le=MAX_BATCH_SIZE)
    epochs: int | None = Field(default=1, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    learning_rate: float = Field(default=3e-5, gt=0)
    max_sequence_length: int = Field(default=512, ge=2)
    gradient_checkpointing: bool | None = None
    num_workers: int = Field(default=2, ge=0)
    weight_decay: float = Field(default=0.01, ge=0)
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    validation_interval: int = Field(default=100, ge=1)
    checkpoint_interval: int = Field(default=100, ge=1)
    validation_batches: int = Field(default=50, ge=0)
    log_interval: int = Field(default=25, ge=1)
    random_seed: int = 42
    max_train_samples: int | None = Field(default=None, ge=1)
    max_validation_samples: int | None = Field(default=None, ge=1)
    mixed_precision: bool = True
    device: Literal["cuda", "cpu"] = "cuda"
    pretrained_engine: Literal["transformers", "unsloth"] = "transformers"
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)

    @model_validator(mode="after")
    def duration(self) -> Self:
        """Epochs и max_steps не могут одновременно ограничивать один запуск."""

        if (self.epochs is None) == (self.max_steps is None):
            raise ValueError("Выберите epochs или max_steps")
        return self


class ManagedTrainingRequest(BaseModel):
    """Выбор модели и версии датасета вместо ручного задания каталогов."""

    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    dataset_version: str
    model_id: str
    parameters: TrainingParameters = Field(default_factory=TrainingParameters)
    parent_run_id: str | None = None
    checkpoint_name: str = "last.pt"


def validate_splits(dataset: DatasetVersion) -> dict[str, str]:
    """Проверить источник и отсутствие thread leakage перед новым запуском."""

    if fingerprint(dataset.directory / "samples.jsonl") != dataset.sha256:
        raise ValueError("Samples версии изменились после публикации; создайте новую версию")
    owners: dict[tuple[str, int], str] = {}
    remaining = Counter(
        sample for _, sample in iter_dialogue_jsonl(dataset.directory / "samples.jsonl")
    )
    hashes = {}
    for split in ("train", "validation", "test"):
        path = dataset.directory / "splits" / f"{split}.jsonl"
        if not path.is_file():
            raise ValueError("Сначала подготовьте splits выбранной версии датасета")
        count = 0
        for _, sample in iter_dialogue_jsonl(path):
            if remaining[sample] <= 0:
                raise ValueError(
                    f"Split {split} содержит sample не из выбранной версии или дубликат"
                )
            remaining[sample] -= 1
            count += 1
            key = (sample.board, sample.thread_id)
            if owners.setdefault(key, split) != split:
                raise ValueError("Обнаружен thread leakage между splits")
        if split != "test" and count == 0:
            raise ValueError(f"Split {split} пуст")
        hashes[split] = fingerprint(path)
    if any(remaining.values()):
        raise ValueError("Splits не покрывают все samples выбранной версии")
    return hashes


def prepare_version_splits(dataset_id: str, version: str) -> dict[str, object]:
    """Создать splits только один раз, не перезаписывая существующие файлы."""

    dataset = get_dataset(dataset_id, version)
    destination = dataset.directory / "splits"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Splits уже существуют; перезапись опубликованной версии запрещена")
    if fingerprint(dataset.directory / "samples.jsonl") != dataset.sha256:
        raise ValueError("Samples версии изменились")
    return split_dataset(SplitConfig(dataset.directory / "samples.jsonl", destination))


def validate_training_request(request: ManagedTrainingRequest) -> None:
    """Выполнить проверки до помещения задачи в очередь и загрузки модели."""

    dataset = get_dataset(request.dataset_id, request.dataset_version)
    hashes = validate_splits(dataset)
    if request.model_id != "custom":
        if load_registered_model(request.model_id) is None:
            raise ValueError("Сначала выполните pretrained setup выбранной модели")
    elif request.parameters.max_sequence_length != ModelConfig().max_sequence_length:
        raise ValueError("Для Custom sequence length определяется ModelConfig")
    if request.parent_run_id:
        parent = get_run(request.parent_run_id)
        if parent is None or (parent.model_id, parent.dataset_id, parent.dataset_version) != (
            request.model_id,
            request.dataset_id,
            request.dataset_version,
        ):
            raise ValueError("Resume требует ту же модель и ту же версию датасета")
        for name in ("batch_size", "random_seed", "max_train_samples"):
            if parent.training_config.get(name) != getattr(request.parameters, name):
                raise ValueError(f"Для resume сохраните исходный {name}")
        if parent.training_config.get("split_sha256", hashes) != hashes:
            raise ValueError("Splits изменились после предыдущего запуска")
        from mini_llm.inference.checkpoints import resolve_named_checkpoint

        resolve_named_checkpoint(parent.directory, request.checkpoint_name)
        if request.model_id != "custom":
            from mini_llm.training.sft.checkpoints import resolve_sft_resume

            resolve_sft_resume(resolve_named_checkpoint(parent.directory, request.checkpoint_name))
            if parent.training_config.get("learning_rate") != request.parameters.learning_rate:
                raise ValueError("Native TRL resume требует исходный learning_rate")
            if (
                parent.training_config.get("pretrained_engine", "transformers")
                != request.parameters.pretrained_engine
            ):
                raise ValueError("При resume сохраните исходный pretrained_engine")
        if parent.pretrained_config_file:
            original = PretrainedConfig.load(parent.pretrained_config_file)
            if (request.parameters.lora_rank, request.parameters.lora_alpha) != (
                original.lora_rank,
                original.lora_alpha,
            ):
                raise ValueError("При resume сохраните исходные LoRA rank/alpha")
            if request.parameters.max_sequence_length != original.max_sequence_length:
                raise ValueError("При resume сохраните исходный max_sequence_length")
            if request.parameters.gradient_checkpointing not in (
                None,
                original.gradient_checkpointing,
            ):
                raise ValueError("При resume сохраните исходный gradient_checkpointing")


def train_versioned_config(
    config: TrainingConfig,
    *,
    job_id: str | None = None,
    pretrained_config: PretrainedConfig | None = None,
) -> TrainingResult:
    """Сохранить любой настроенный запуск в новой версии каталога, не меняя trainer."""

    from mini_llm.training.runtime import training_config_payload

    dataset = next(
        (
            item
            for item in list_datasets()
            if (item.directory / "splits").resolve() == config.splits_dir.resolve()
        ),
        None,
    )
    if dataset is None:
        raise ValueError(
            "Splits не относятся к опубликованной версии; используйте workspace publish"
        )
    hashes = validate_splits(dataset)
    base = pretrained_config or (
        PretrainedConfig.load(config.pretrained_config_file)
        if config.pretrained_config_file
        else None
    )
    model_id = base.model_id if base else "custom"
    parent = None
    if config.resume_from:
        parent = next(
            (
                item
                for item in list_runs()
                if item.directory.resolve() == config.resume_from.parent.resolve()
            ),
            None,
        )
        if parent is None or (parent.model_id, parent.dataset_id, parent.dataset_version) != (
            model_id,
            dataset.dataset_id,
            dataset.version,
        ):
            raise ValueError("Resume требует checkpoint той же модели и версии датасета")
        if parent.training_config.get("split_sha256", hashes) != hashes:
            raise ValueError("Splits изменились после предыдущего запуска")
        for name in ("batch_size", "random_seed", "max_train_samples"):
            if parent.training_config.get(name) != getattr(config, name):
                raise ValueError(f"Для resume сохраните исходный {name}")
        if (
            base
            and parent.pretrained_config_file
            and PretrainedConfig.load(parent.pretrained_config_file) != base
        ):
            raise ValueError("При resume нельзя менять pretrained-конфиг")
    slug = ModelPaths.for_model(model_id).config_file.parent.name if base else "custom"
    run = reserve_run(slug, model_id, dataset)
    run.status = "running"
    run.job_id = job_id
    run.parent_run_id = parent.run_id if parent else None
    run.tokenizer_file = config.tokenizer_file if base is None else None
    run.save()
    try:
        if base:
            run.pretrained_config_file = run.directory / "pretrained.json"
            base.save(run.pretrained_config_file)
        active = replace(
            config, checkpoint_dir=run.directory, pretrained_config_file=run.pretrained_config_file
        )
        run.training_config = training_config_payload(active)
        if base:
            run.training_config.update(
                trainer="trl",
                max_sequence_length=base.max_sequence_length,
                gradient_checkpointing=base.gradient_checkpointing,
                lora_rank=base.lora_rank,
                lora_alpha=base.lora_alpha,
            )
        run.training_config["split_sha256"] = hashes
        run.save()
        result = train_model(active)
        run = ModelRun.model_validate_json((run.directory / "run.json").read_text(encoding="utf-8"))
        run.global_step = result.global_step
        run.status = "interrupted" if result.interrupted else "completed"
        run.save()
        return result
    except Exception as error:
        run = ModelRun.model_validate_json((run.directory / "run.json").read_text(encoding="utf-8"))
        run.status = "failed"
        run.error = f"{type(error).__name__}: {error}"
        run.save()
        raise


def run_managed_training(request: ManagedTrainingRequest, job_id: str) -> dict[str, object]:
    """Преобразовать выбор UI в прежний TrainingConfig и изолированный новый запуск."""

    from mini_llm.inference.checkpoints import resolve_named_checkpoint

    validate_training_request(request)
    dataset = get_dataset(request.dataset_id, request.dataset_version)
    parent = get_run(request.parent_run_id) if request.parent_run_id else None
    registration = load_registered_model(request.model_id) if request.model_id != "custom" else None
    values = request.parameters.model_dump(
        exclude={"max_sequence_length", "gradient_checkpointing", "lora_rank", "lora_alpha"}
    )
    values.update(
        splits_dir=dataset.directory / "splits",
        token_statistics_file=dataset.directory / "reports/token_statistics.json",
        tokenizer_file=(parent.tokenizer_file if parent else None) or DEFAULT_TOKENIZER_PATH,
    )
    if parent:
        values["resume_from"] = resolve_named_checkpoint(parent.directory, request.checkpoint_name)
    base = None
    active: TrainingConfig
    if registration:
        path = parent.pretrained_config_file if parent else registration.config_file
        if path is None:
            raise ValueError("Не найден конфиг модели")
        base = PretrainedConfig.load(path).model_copy(
            update={
                "max_sequence_length": request.parameters.max_sequence_length,
                "lora_rank": request.parameters.lora_rank,
                "lora_alpha": request.parameters.lora_alpha,
            }
        )
        if request.parameters.gradient_checkpointing is not None:
            base = base.model_copy(
                update={"gradient_checkpointing": request.parameters.gradient_checkpointing}
            )
        active = PretrainedTrainingConfig(pretrained_config_file=path, **values)
    else:
        active = TrainingConfig(**values)
    result = train_versioned_config(active, job_id=job_id, pretrained_config=base)
    run = next(item for item in list_runs() if item.directory == result.last_checkpoint.parent)
    return {
        "run_id": run.run_id,
        "checkpoint_dir": run.directory.as_posix(),
        "global_step": result.global_step,
        "interrupted": result.interrupted,
    }
