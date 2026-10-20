"""Точечная диагностика завершения ответов без изменения модели и рабочих настроек."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import re
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from torch import Tensor
from transformers import AutoTokenizer, RepetitionPenaltyLogitsProcessor, TopKLogitsWarper

from mini_llm.data.dialogue import iter_dialogue_jsonl
from mini_llm.data.interfaces import OversizedResponseError
from mini_llm.inference.service import _checkpoint_model_state, _read_checkpoint
from mini_llm.pretrained import PretrainedConfig, prepare_pretrained_model
from mini_llm.pretrained.inference import QWEN_SYSTEM_PROMPT, _stop_token_ids
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer
from mini_llm.training.sft.data import ResponseOnlyCollator, prepare_dataset


def save(path: Path, report: dict[str, object]) -> None:
    """Сохранять частичные измерения после каждого этапа."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def examples(path: Path, checkpoint: str) -> list[dict[str, str]]:
    """Взять три ответа и вручную выбранные границы до первого мусорного фрагмента."""

    sections = re.split(r"(?m)^## ", path.read_text(encoding="utf-8"))
    markers = {1: "ストレス", 2: "놧", 3: "新人玩家"}
    result = []
    for section in sections:
        if not section.startswith(checkpoint + " · вопрос "):
            continue
        number = int(section.splitlines()[0].rsplit(" ", 1)[1])
        if number not in markers:
            continue
        blocks = re.findall(r"(?ms)^(`{3,})text\n(.*?)\n\1$", section)
        if len(blocks) < 2:
            raise ValueError("Не удалось прочитать вопрос/ответ в отчёте")
        question, answer = blocks[0][1], blocks[1][1]
        marker = markers[number]
        if marker not in answer:
            raise ValueError(f"Не найдена вручную выбранная граница для вопроса {number}")
        result.append(
            {"question": question, "answer": answer, "prefix": answer.split(marker, 1)[0]}
        )
    if len(result) != 3:
        raise ValueError("Нужны три сохранённых ответа выбранного checkpoint")
    return result


def token_stats(scores: Tensor, token_id: int) -> dict[str, object]:
    """Вернуть вероятность и строгий rank токена в полном словаре."""

    return {
        "id": token_id,
        "probability": float(scores.softmax(-1)[0, token_id].item()),
        "rank": int((scores[0] > scores[0, token_id]).sum().item()) + 1,
    }


def distribution(
    scores: Tensor, tokenizer: HuggingFaceDialogueTokenizer, stops: list[int]
) -> dict[str, object]:
    """Измерить stop-токены и TOP-10 без округления малых вероятностей."""

    probabilities = scores.softmax(-1)
    ids = scores[0].topk(10).indices.tolist()
    return {
        "stop_tokens": [
            {
                "token": tokenizer.decode([item], skip_special_tokens=False),
                **token_stats(scores, item),
            }
            for item in stops
        ],
        "top_10": [
            {
                "id": item,
                "text": tokenizer.decode([item], skip_special_tokens=False),
                "probability": float(probabilities[0, item].item()),
            }
            for item in ids
        ],
    }


def audit_labels(
    train_file: Path, tokenizer: HuggingFaceDialogueTokenizer, max_length: int, end_id: int
) -> dict[str, object]:
    """Проверить реальные labels до и после штатного collator на CPU."""

    rows = prepare_dataset(train_file, tokenizer, max_length=max_length, limit=None)
    supervised = total = 0
    selected: list[dict[str, object]] = []
    collator = ResponseOnlyCollator(tokenizer.pad_token_id)
    for index in range(len(rows)):
        row = rows[index]
        positions = [i for i, value in enumerate(row["input_ids"]) if value == end_id]
        response_positions = [i for i in positions if row["labels"][i] == end_id]
        total += len(response_positions)
        supervised += bool(response_positions)
        if len(selected) < 5:
            batch = collator([row])
            selected.append(
                {
                    "sample_index": index,
                    "length": len(row["input_ids"]),
                    "response_end_positions": response_positions,
                    "labels_after_collator": [
                        int(batch["labels"][0, i].item()) for i in response_positions
                    ],
                    "causal_previous_positions": [i - 1 for i in response_positions],
                    "tail_ids": row["input_ids"][-12:],
                    "tail_text": tokenizer.decode(
                        row["input_ids"][-12:], skip_special_tokens=False
                    ),
                }
            )
    scanned = oversized = 0
    for _, sample in iter_dialogue_jsonl(train_file):
        scanned += 1
        try:
            tokenizer.encode_training_window(sample, max_length=max_length)
        except OversizedResponseError:
            oversized += 1
    return {
        "raw_samples": scanned,
        "usable_samples": len(rows),
        "oversized_skipped_whole": oversized,
        "samples_with_supervised_im_end": supervised,
        "supervised_im_end_tokens": total,
        "real_examples": selected,
    }


def audit_embeddings(base: PretrainedConfig, token_id: int) -> dict[str, object]:
    """Проверить различимость строки конца сообщения в локальных исходных весах."""

    weights = (
        base.cache_dir
        / f"models--{base.model_id.replace('/', '--')}"
        / "snapshots"
        / str(base.revision)
        / "model.safetensors"
    )
    with safe_open(weights, framework="pt", device="cpu") as handle:
        matrix = handle.get_slice("model.embed_tokens.weight")
        row = matrix[token_id : token_id + 1]
        same_ids: list[int] = []
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            for start in range(0, matrix.get_shape()[0], 2048):
                matches = (matrix[start : start + 2048] == row).all(dim=1)
                same_ids.extend((matches.nonzero().flatten() + start).tolist())
        finally:
            torch.set_num_threads(previous_threads)
        return {
            "file": str(weights),
            "im_end_norm": float(row.float().norm()),
            "exactly_identical_rows": len(same_ids),
            "identical_ids": same_ids,
            "im_start_equal_im_end": torch.equal(row, matrix[151644:151645]),
        }


def main() -> None:
    """Один раз загрузить локальные веса, снять logits и сравнить два sampling-режима."""

    cast(io.TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("checkpoints/runs/model--qwen--qwen3-1.7b-base/2ch/v5/run-002"),
    )
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    base = PretrainedConfig.load(args.run_dir / "pretrained.json")
    if not base.local_files_only:
        raise ValueError("Диагностика требует local_files_only=true, без скачивания весов")
    checkpoint = max(args.run_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    selected = examples(args.source_report, checkpoint.name)
    train_file = Path(run["training_config"]["splits_dir"]) / "train.jsonl"
    report: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "source_report": str(args.source_report),
        "train_file": str(train_file),
        "train_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
        "training_config": run["training_config"],
        "pretrained_config": base.model_dump(mode="json"),
        "boundary_note": (
            "Ручные границы; token IDs восстановлены из текста, исходные IDs не сохранены."
        ),
    }
    tokenizer = HuggingFaceDialogueTokenizer(
        AutoTokenizer.from_pretrained(
            base.model_id,
            revision=base.revision,
            cache_dir=str(base.cache_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
    )
    end_id = tokenizer.encode("<|im_end|>")[0]
    stops = _stop_token_ids(tokenizer, is_qwen=True)
    report["labels"] = audit_labels(train_file, tokenizer, base.max_sequence_length, end_id)
    saved_tokenizer = HuggingFaceDialogueTokenizer(
        AutoTokenizer.from_pretrained(
            checkpoint.parent / f"checkpoint-{int(checkpoint.stem.split('_')[1])}",
            revision=base.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
    )
    report["tokenizer_comparison"] = {
        "same_prompt_ids": all(
            tokenizer.encode_prompt(item["question"], system_prompt=QWEN_SYSTEM_PROMPT)
            == saved_tokenizer.encode_prompt(item["question"], system_prompt=QWEN_SYSTEM_PROMPT)
            for item in selected
        ),
        "im_end_id": end_id,
        "saved_im_end_ids": saved_tokenizer.encode("<|im_end|>"),
        "eos_id": tokenizer.eos_token_id,
    }
    report["embedding_audit"] = audit_embeddings(base, end_id)
    save(args.output, report)
    print("Labels и tokenizer проверены", report["labels"], flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Для измерения logits требуется свободная CUDA GPU")
    free, _ = torch.cuda.mem_get_info()
    if free < 3 * 1024**3:
        raise RuntimeError("Освободите минимум 3 GiB VRAM: остановите backend/обучение")
    prepared = prepare_pretrained_model(base, for_inference=True)
    payload = _read_checkpoint(checkpoint)
    if payload.get("model_metadata") != prepared.model.checkpoint_metadata:
        raise ValueError("Checkpoint несовместим с конфигурацией")
    prepared.model.load_checkpoint_state_dict(_checkpoint_model_state(payload))
    del payload
    prepared.model.eval()
    module = prepared.model.module
    report["embedding_trainability"] = {
        "input_embeddings_frozen": all(
            not value.requires_grad
            for name, value in module.named_parameters()
            if "embed_tokens" in name
        ),
        "lm_head_frozen": all(
            not value.requires_grad
            for name, value in module.named_parameters()
            if "lm_head" in name
        ),
        "embedding_or_head_adapter_parameters": [
            name
            for name, value in module.named_parameters()
            if value.requires_grad and ("embed_tokens" in name or "lm_head" in name)
        ],
    }
    device = next(module.parameters()).device
    report["device_map"] = str(getattr(module, "hf_device_map", None))
    measurements: list[dict[str, object]] = []
    report["boundaries"] = measurements
    with torch.inference_mode():
        training_checks: list[dict[str, object]] = []
        report["training_boundaries"] = training_checks
        for sample_index, (_, sample) in enumerate(iter_dialogue_jsonl(train_file)):
            if sample_index >= 5:
                break
            encoded = tokenizer.encode_training_window(sample, max_length=base.max_sequence_length)
            position = max(i for i, value in enumerate(encoded.token_ids) if value == end_id)
            ids = torch.tensor([encoded.token_ids[:position]], device=device)
            call = cast(Callable[..., object], module)
            output = call(input_ids=ids, use_cache=False, logits_to_keep=1)
            logits = getattr(output, "logits", None)
            if not isinstance(logits, Tensor):
                raise RuntimeError("Модель не вернула logits")
            scores = logits[:, -1, :].float()
            training_checks.append(
                {
                    "sample_index": sample_index,
                    "im_end": token_stats(scores, end_id),
                    "im_start": token_stats(scores, 151644),
                    "stop_cross_entropy": float(-scores.log_softmax(-1)[0, end_id].item()),
                }
            )
            del ids, output, logits, scores
        save(args.output, report)
        for item in selected:
            prompt_ids = tokenizer.encode_prompt(item["question"], system_prompt=QWEN_SYSTEM_PROMPT)
            full_answer = tokenizer.encode(item["answer"])
            prefix_ids = tokenizer.encode(item["prefix"])
            common = 0
            for left, right in zip(prefix_ids, full_answer, strict=False):
                if left != right:
                    break
                common += 1
            prefix_ids = prefix_ids[:common]
            input_ids = torch.tensor([prompt_ids + prefix_ids], device=device)
            call_model = cast(Callable[..., object], module)
            output = call_model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=1,
            )
            logits = getattr(output, "logits", None)
            if not isinstance(logits, Tensor):
                raise RuntimeError("Модель не вернула logits")
            scores = logits[:, -1, :].float()
            penalized = RepetitionPenaltyLogitsProcessor(1.1)(input_ids, scores.clone())
            scaled = penalized / 0.3
            filtered = TopKLogitsWarper(20)(input_ids, scaled.clone())
            observed_id = full_answer[common]
            measurement: dict[str, object] = {
                "question": item["question"],
                "logical_prefix": item["prefix"],
                "measured_prefix": tokenizer.decode(prefix_ids, skip_special_tokens=False),
                "original_continuation": item["answer"][len(item["prefix"]) :][:180],
                "observed_next_token": {
                    "id": observed_id,
                    "text": tokenizer.decode([observed_id], skip_special_tokens=False),
                    "in_current_top_k": bool(torch.isfinite(filtered[0, observed_id]).item()),
                },
                "raw": distribution(scores, tokenizer, stops),
                "after_repetition_penalty": distribution(penalized, tokenizer, stops),
                "after_temperature": distribution(scaled, tokenizer, stops),
                "after_top_k": distribution(filtered, tokenizer, stops),
            }
            measurements.append(measurement)
            save(args.output, report)
            print(json.dumps(measurement, ensure_ascii=False), flush=True)
            del output, logits, scores, penalized, scaled, filtered, input_ids
        generations: list[dict[str, object]] = []
        report["generation_comparison"] = generations
        if not args.skip_generation:
            prompt = selected[0]["question"]
            prompt_ids = tokenizer.encode_prompt(prompt, system_prompt=QWEN_SYSTEM_PROMPT)
            for top_k in (20, 0):
                print(f"Генерация: top_k={top_k}, max_new_tokens=1024", flush=True)
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                ids = torch.tensor([prompt_ids], device=device)
                generate = cast(Callable[..., Tensor], module.generate)
                started = time.perf_counter()
                generated = generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=1024,
                    temperature=0.3,
                    top_k=top_k,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=stops,
                )
                new_ids = cast(list[int], generated[0, len(prompt_ids) :].tolist())
                generations.append(
                    {
                        "question": prompt,
                        "top_k": top_k,
                        "temperature": 0.3,
                        "repetition_penalty": 1.1,
                        "seed": 42,
                        "elapsed_seconds": time.perf_counter() - started,
                        "generated_tokens": len(new_ids),
                        "token_ids": new_ids,
                        "ended_with_stop": bool(new_ids and new_ids[-1] in stops),
                        "last_token_id": new_ids[-1] if new_ids else None,
                        "text_with_special_tokens": tokenizer.decode(
                            new_ids, skip_special_tokens=False
                        ),
                    }
                )
                save(args.output, report)
                print(
                    f"Готово: top_k={top_k}, tokens={len(new_ids)}, "
                    f"stop={generations[-1]['ended_with_stop']}",
                    flush=True,
                )
                del generated, ids
                gc.collect()
    save(args.output, report)
    print(f"Диагностика сохранена: {args.output}", flush=True)


if __name__ == "__main__":
    main()
