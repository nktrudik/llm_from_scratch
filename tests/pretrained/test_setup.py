"""Setup и расположение артефактов на подменённом Hub, без скачивания моделей."""

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mini_llm.pretrained import PretrainedConfig, cli, download, setup
from mini_llm.pretrained.config import AdaptationMode
from mini_llm.pretrained.download import DownloadedSnapshot, select_model_files
from mini_llm.pretrained.workspace import ACTIVE_MODEL_FILE, ModelPaths, ModelRegistration
from mini_llm.training import TrainingConfig
from mini_llm.training.schemas import TrainingResult


@pytest.fixture
def fake_setup_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Подменить Hub и AutoTokenizer, сохранив настоящую запись config/registration."""

    monkeypatch.chdir(tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    def fake_download(model_id: str, revision: str, cache_dir: Path) -> DownloadedSnapshot:
        return DownloadedSnapshot(snapshot, "a" * 40)

    module = ModuleType("transformers")
    module.__dict__["AutoConfig"] = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(is_encoder_decoder=False)
    )
    module.__dict__["AutoTokenizer"] = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(
            eos_token_id=2, eos_token="eos", pad_token_id=0
        )
    )
    monkeypatch.setattr(setup, "download_model_snapshot", fake_download)
    monkeypatch.setattr(setup, "require_module", lambda name: module)
    return snapshot


@pytest.mark.parametrize("mode", ["full", "lora", "qlora"])
def test_setup_pins_revision_and_registers_artifacts(
    fake_setup_environment: Path, mode: AdaptationMode
) -> None:
    registration = setup.setup_pretrained_model("owner/model", mode=mode)
    paths = ModelPaths.for_model("owner/model", mode)
    config = PretrainedConfig.load(registration.config_file)

    assert config.model_id == "owner/model"
    assert config.revision == "a" * 40
    assert config.adaptation_mode == mode
    assert config.local_files_only
    assert config.max_sequence_length == 512
    assert config.torch_dtype == ("float32" if mode == "full" else "bfloat16")
    assert config.gradient_checkpointing == (mode != "qlora")
    assert config.lora_rank == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert config.lora_target_modules == "all-linear"
    assert config.qlora_quant_type == "nf4"
    assert config.qlora_double_quant
    assert registration.config_file == paths.config_file
    assert registration.snapshot_path == fake_setup_environment
    assert registration.checkpoint_dir.is_dir()
    assert not registration.best_checkpoint.exists()
    assert ModelRegistration.model_validate_json(ACTIVE_MODEL_FILE.read_text()) == registration


def test_setup_does_not_overwrite_config_of_existing_checkpoints(
    fake_setup_environment: Path,
) -> None:
    registration = setup.setup_pretrained_model("owner/model")
    old_config = PretrainedConfig.load(registration.config_file).model_copy(update={"lora_rank": 8})
    old_config.save(registration.config_file)
    registration.best_checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(RuntimeError, match="не перезаписан"):
        setup.setup_pretrained_model("owner/model")
    assert PretrainedConfig.load(registration.config_file).lora_rank == 8


@pytest.mark.parametrize("model_id", ["../model", "a/../../model", "/tmp/model", "a\\b", "a//b"])
def test_model_paths_reject_traversal(model_id: str) -> None:
    with pytest.raises(ValueError, match="model ID"):
        ModelPaths.for_model(model_id)


def test_model_paths_isolate_models_and_adaptation_modes() -> None:
    first = ModelPaths.for_model("owner/model", "lora")
    second = ModelPaths.for_model("owner/model", "qlora")
    another = ModelPaths.for_model("another/model", "lora")
    assert len({first.checkpoint_dir, second.checkpoint_dir, another.checkpoint_dir}) == 3
    assert len({first.config_file, second.config_file, another.config_file}) == 3


def test_download_prefers_safetensors_and_retains_tokenizer_assets() -> None:
    files = [
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "chat_template.jinja",
        "onnx/model.onnx",
        "onnx/config.json",
        "training_args.bin",
        "README.md",
    ]
    assert select_model_files(files) == [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "chat_template.jinja",
    ]


def test_download_supports_bin_fallback_and_nested_shards() -> None:
    assert "pytorch_model.bin" in select_model_files(["config.json", "pytorch_model.bin"])
    assert "weights/part.safetensors" in select_model_files(
        ["config.json", "model.safetensors.index.json", "weights/part.safetensors"]
    )


def test_download_pins_snapshot_to_hub_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    module = ModuleType("huggingface_hub")
    module.__dict__["HfApi"] = lambda: SimpleNamespace(
        model_info=lambda *args, **kwargs: SimpleNamespace(
            sha="b" * 40,
            siblings=[
                SimpleNamespace(rfilename=name) for name in ("config.json", "model.safetensors")
            ],
        )
    )
    module.__dict__["snapshot_download"] = snapshot_download
    monkeypatch.setattr(download, "require_module", lambda name: module)

    result = download.download_model_snapshot("owner/model", "main", tmp_path / "cache")
    assert result.revision == "b" * 40
    assert calls[0]["revision"] == "b" * 40
    assert calls[0]["allow_patterns"] == ["config.json", "model.safetensors"]


def test_setup_cli_requires_only_model_id() -> None:
    args = cli.build_argument_parser().parse_args(["setup", "owner/model"])
    assert args.command == "setup"
    assert args.mode == "qlora"
    assert args.batch_size == 2
    assert args.max_sequence_length == 512
    assert not args.train


def test_setup_cli_does_not_train_without_flag(
    fake_setup_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mini_llm.training as training

    def unexpected_training(*args: object, **kwargs: object) -> None:
        pytest.fail("Setup без --train не должен запускать обучение")

    monkeypatch.setattr(training, "train_model", unexpected_training)
    assert cli.main(["setup", "owner/model"]) == 0


def test_setup_cli_train_requires_existing_splits_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def unexpected_download(*args: object, **kwargs: object) -> None:
        pytest.fail("Если split отсутствует, не нужно скачивать модель")

    monkeypatch.setattr(cli, "setup_pretrained_model", unexpected_download)
    assert cli.main(["setup", "owner/model", "--train"]) == 1


def test_setup_cli_train_passes_registered_paths_to_existing_trainer(
    fake_setup_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mini_llm.training as training

    splits = Path("data/processed/splits")
    splits.mkdir(parents=True)
    for split in ("train", "validation"):
        (splits / f"{split}.jsonl").touch()
    calls: list[TrainingConfig] = []

    def train(config: TrainingConfig) -> TrainingResult:
        calls.append(config)
        return TrainingResult(1, 1.0, config.checkpoint_dir / "last.pt", False)

    monkeypatch.setattr(training, "train_model", train)
    assert cli.main(["setup", "owner/model", "--train"]) == 0
    assert len(calls) == 1
    assert calls[0].model_backend == "pretrained"
    assert calls[0].pretrained_config_file == ModelPaths.for_model("owner/model").config_file
    assert calls[0].checkpoint_dir == ModelPaths.for_model("owner/model").checkpoint_dir
    assert calls[0].batch_size == 2
    assert calls[0].max_train_samples == 30_000
    assert calls[0].num_workers == 2
    assert calls[0].log_interval == 100
    assert calls[0].validation_interval == 500
    assert calls[0].validation_batches == 50
    assert calls[0].checkpoint_interval == 500
    assert calls[0].epochs == 3
    assert calls[0].learning_rate == 0.0002


def test_setup_does_not_activate_failed_download(
    fake_setup_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> DownloadedSnapshot:
        raise RuntimeError("Ошибка Hub")

    monkeypatch.setattr(setup, "download_model_snapshot", fail)
    with pytest.raises(RuntimeError, match="Ошибка Hub"):
        setup.setup_pretrained_model("owner/model")
    assert not ACTIVE_MODEL_FILE.exists()


def test_setup_cli_preserves_explicit_fp16_and_full_dataset_option(
    fake_setup_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mini_llm.training as training

    splits = Path("data/processed/splits")
    splits.mkdir(parents=True)
    for split in ("train", "validation"):
        (splits / f"{split}.jsonl").touch()
    calls: list[TrainingConfig] = []

    def train(config: TrainingConfig) -> TrainingResult:
        calls.append(config)
        return TrainingResult(1, 1.0, config.checkpoint_dir / "last.pt", False)

    monkeypatch.setattr(training, "train_model", train)
    assert (
        cli.main(
            [
                "setup",
                "owner/model",
                "--train",
                "--dtype",
                "float16",
                "--gradient-checkpointing",
                "--max-train-samples",
                "0",
                "--num-workers",
                "0",
                "--batch-size",
                "1",
                "--log-interval",
                "7",
                "--validation-interval",
                "13",
                "--validation-batches",
                "0",
                "--checkpoint-interval",
                "17",
            ]
        )
        == 0
    )
    config = PretrainedConfig.load(ModelPaths.for_model("owner/model").config_file)
    assert config.torch_dtype == "float16"
    assert config.gradient_checkpointing
    assert config.max_sequence_length == 512
    assert calls[0].max_train_samples is None
    assert calls[0].batch_size == 1
    assert calls[0].num_workers == 0
    assert calls[0].log_interval == 7
    assert calls[0].validation_interval == 13
    assert calls[0].validation_batches == 0
    assert calls[0].checkpoint_interval == 17
