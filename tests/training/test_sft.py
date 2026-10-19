"""Регрессии нового SFT на маленькой случайной CPU-модели без Hub и GPU."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import cast

import pytest
import torch
from torch import Tensor

from mini_llm.api.jobs import JobManager
from mini_llm.data.dialogue import parse_dialogue_sample
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.model import HuggingFaceCausalLMBackend, PreparedPretrained
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer
from mini_llm.training.config import PretrainedTrainingConfig
from mini_llm.training.control import training_control, training_stop_requested
from mini_llm.training.pipeline import train_model
from mini_llm.training.sft import pipeline
from mini_llm.training.sft.checkpoints import resolve_sft_resume
from mini_llm.training.sft.config import sft_arguments
from mini_llm.training.sft.data import ResponseOnlyCollator, prepare_dataset


@pytest.fixture
def sft_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> PretrainedTrainingConfig:
    """Создать только синтетические сообщения и случайные веса на CPU."""

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    backend = Tokenizer(
        WordLevel(
            {
                "<unk>": 0,
                "<pad>": 1,
                "<eos>": 2,
                "user": 3,
                "assistant": 4,
                "hello": 5,
                "answer": 6,
                "question": 7,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    native = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    native.chat_template = (
        "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' <eos> ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    )
    base = PretrainedConfig(
        model_id="local/tiny",
        adaptation_mode="lora" if getattr(request, "param", "full") == "lora" else "full",
        torch_dtype="float32",
        gradient_checkpointing=False,
        max_sequence_length=32,
    )
    base.save(tmp_path / "pretrained.json")
    samples = [
        {
            "messages": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "answer"},
            ]
        }
        for text in ("hello question", "question hello", "question question", "hello hello")
    ]
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8"
        )

    def prepare(config: PretrainedConfig) -> PreparedPretrained:
        model = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=len(native),
                n_positions=32,
                n_embd=8,
                n_layer=1,
                n_head=1,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                eos_token_id=2,
                pad_token_id=1,
            )
        )
        if config.adaptation_mode == "lora":
            peft = pytest.importorskip("peft")
            model = peft.get_peft_model(
                model,
                peft.LoraConfig(
                    task_type="CAUSAL_LM",
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=["c_attn"],
                ),
            )
        return PreparedPretrained(
            HuggingFaceCausalLMBackend(model, config), HuggingFaceDialogueTokenizer(native)
        )

    monkeypatch.setattr(pipeline, "prepare_pretrained_model", prepare)
    return PretrainedTrainingConfig(
        splits_dir=tmp_path,
        checkpoint_dir=tmp_path / "first",
        pretrained_config_file=tmp_path / "pretrained.json",
        batch_size=2,
        epochs=None,
        max_steps=1,
        device="cpu",
        num_workers=0,
        mixed_precision=False,
        validation_interval=1,
        checkpoint_interval=1,
        log_interval=1,
        max_train_samples=None,
        max_validation_samples=None,
    )


@pytest.mark.parametrize("sft_setup", ["full", "lora"], indirect=True)
def test_sft_native_checkpoint_resume_and_inference_export(
    sft_setup: PretrainedTrainingConfig,
) -> None:
    first = train_model(sft_setup)
    original = first.last_checkpoint.read_bytes()
    native = resolve_sft_resume(first.last_checkpoint)
    assert (native / "optimizer.pt").is_file()
    second = train_model(
        replace(
            sft_setup,
            checkpoint_dir=sft_setup.checkpoint_dir.parent / "second",
            resume_from=first.last_checkpoint,
            max_steps=3,
        )
    )
    assert second.global_step == 3
    assert first.last_checkpoint.read_bytes() == original
    state = torch.load(second.last_checkpoint, weights_only=True)
    assert state["samples_seen"] == 6
    assert state["tokens_seen"] > 0
    assert state["model_metadata"]["backend"] == "pretrained"
    assert state["model_state_dict"]
    assert second.last_checkpoint.with_name("best.pt").is_file()
    uninterrupted = train_model(
        replace(
            sft_setup, checkpoint_dir=sft_setup.checkpoint_dir.parent / "uninterrupted", max_steps=3
        )
    )
    expected = torch.load(uninterrupted.last_checkpoint, weights_only=True)["model_state_dict"]
    for name, value in cast(dict[str, Tensor], state["model_state_dict"]).items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_sft_response_only_collator_and_chatml(sft_setup: PretrainedTrainingConfig) -> None:
    assert sft_setup.pretrained_config_file is not None
    base = PretrainedConfig.load(sft_setup.pretrained_config_file)
    prepare = cast(
        Callable[[PretrainedConfig], PreparedPretrained],
        pipeline.__dict__["prepare_pretrained_model"],
    )
    prepared = prepare(base)
    data = prepare_dataset(
        sft_setup.splits_dir / "train.jsonl", prepared.tokenizer, max_length=32, limit=1
    )
    row = data[0]
    prompt = prepared.tokenizer.encode_prompt("hello question")
    assert row["input_ids"][: len(prompt)] == prompt
    assert row["labels"][: len(prompt)] == [-100] * len(prompt)
    assert row["labels"][len(prompt) :] == row["input_ids"][len(prompt) :]
    short = {"input_ids": [3, 6], "labels": [-100, 6]}
    batch = ResponseOnlyCollator(1)([row, short])
    assert batch["labels"][1, 2:].eq(-100).all()
    assert batch["attention_mask"][1, 2:].eq(0).all()


def test_chatml_preserves_content_system_and_thread() -> None:
    sample = parse_dialogue_sample(
        {
            "messages": [
                {"role": "system", "content": "  Правила\n"},
                {"role": "user", "content": " Вопрос 😄\n"},
                {"role": "assistant", "content": " Ответ!  "},
            ],
            "metadata": {"board": "b", "thread_id": 42},
        }
    )
    assert sample.context == (" Вопрос 😄\n",)
    assert sample.response == " Ответ!  "
    assert sample.messages[0] == ("system", "  Правила\n")
    assert (sample.board, sample.thread_id) == ("b", 42)


def test_legacy_resume_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "last.pt"
    torch.save({"global_step": 200}, path)
    with pytest.raises(ValueError, match="старым trainer"):
        resolve_sft_resume(path)


def test_sft_defaults_and_job_stop() -> None:
    base = PretrainedConfig(model_id="Qwen/Qwen3-0.6B", adaptation_mode="qlora")
    args = sft_arguments(PretrainedTrainingConfig(pretrained_config_file=Path("model.json")), base)
    assert (base.lora_rank, base.lora_alpha) == (8, 16)
    assert args["bf16"] and not args["fp16"]
    assert args["optim"] == "adamw_8bit" and not args["packing"]
    manager = JobManager()
    record = manager.create("training")
    manager.request_stop(record.job_id)
    seen: list[bool] = []
    manager.run(record.job_id, lambda: seen.append(training_stop_requested()))
    assert seen == [True]
    assert not training_stop_requested()


def test_sft_stop_saves_native_state(
    sft_setup: PretrainedTrainingConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mini_llm.training.sft.trainer import SFTObserver

    event = Event()
    original = SFTObserver.on_step_end

    def stop(self: SFTObserver, *args: object, **kwargs: object) -> None:
        event.set()
        original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(SFTObserver, "on_step_end", stop)
    with training_control(event):
        result = train_model(replace(sft_setup, max_steps=5))
    assert result.interrupted and result.global_step == 1
    assert resolve_sft_resume(result.last_checkpoint).is_dir()
