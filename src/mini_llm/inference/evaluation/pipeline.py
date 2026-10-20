"""Последовательная очередь checkpoint → пять независимых вопросов."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import torch

from mini_llm.inference.checkpoints import resolve_named_checkpoint
from mini_llm.inference.config import GenerationConfig
from mini_llm.inference.evaluation.report import (
    REPORTS_ROOT,
    append_answer,
    create_report,
    finish_report,
)
from mini_llm.inference.evaluation.schemas import (
    QUESTIONS,
    EvaluationAnswer,
    EvaluationProgress,
    EvaluationRequest,
)
from mini_llm.inference.evaluation.session import CheckpointSession
from mini_llm.inference.schemas import GenerationResult
from mini_llm.inference.service import clear_pretrained_cache
from mini_llm.observability import terminal_log
from mini_llm.training.control import training_stop_requested
from mini_llm.workspace.catalog import ModelRun, contained, get_run
from mini_llm.workspace.paths import RUNS_ROOT


class EvaluationSession(Protocol):
    """Минимальный контракт для реального inference и быстрых тестовых заглушек."""

    def load(self, checkpoint: Path) -> None: ...
    def generate(self, prompt: str) -> GenerationResult: ...
    def close(self) -> None: ...


def evaluation_plan(request: EvaluationRequest) -> tuple[ModelRun, list[Path]]:
    """Снять список всех .pt checkpoints выбранного запуска без чтения весов."""

    run = get_run(request.run_id)
    if run is None:
        raise ValueError("Запуск не найден")
    contained(run.directory, RUNS_ROOT)
    if run.model_id == "custom":
        if run.tokenizer_file is None or not run.tokenizer_file.is_file():
            raise ValueError("У запуска Custom не найден tokenizer")
    elif run.pretrained_config_file is None or not run.pretrained_config_file.is_file():
        raise ValueError("Не найден pretrained-конфиг выбранного запуска")
    paths = [resolve_named_checkpoint(run.directory, p.name) for p in run.directory.glob("*.pt")]
    if not paths:
        raise ValueError("В выбранном запуске нет checkpoints")
    paths.sort(
        key=lambda p: (
            0 if p.stem.startswith("step_") and p.stem[5:].isdigit() else 1,
            int(p.stem[5:]) if p.stem.startswith("step_") and p.stem[5:].isdigit() else 0,
            p.name,
        )
    )
    return run, paths


def run_evaluation(
    request: EvaluationRequest,
    *,
    report_progress: Callable[[EvaluationProgress], None] | None = None,
    should_stop: Callable[[], bool] = training_stop_requested,
    session_factory: Callable[[ModelRun, GenerationConfig], EvaluationSession] = CheckpointSession,
) -> dict[str, object]:
    """Запускать только явно; сохранять каждый ответ до перехода к следующему вопросу."""

    run, checkpoints = evaluation_plan(request)
    signatures = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in checkpoints}
    output = REPORTS_ROOT / run.run_id / f"evaluation-{uuid4().hex}.md"
    contained(output, REPORTS_ROOT)
    create_report(output, request, run, checkpoints)
    progress = EvaluationProgress(
        total=len(checkpoints) * len(QUESTIONS),
        total_checkpoints=len(checkpoints),
        report_file=output.as_posix(),
    )
    started = time.perf_counter()
    session: EvaluationSession | None = None
    answer_seconds = swap_seconds = 0.0
    loaded_checkpoints = 0

    def publish(stage: str) -> None:
        progress.stage = stage
        progress.elapsed_seconds = time.perf_counter() - started
        if progress.completed:
            # Однократную загрузку base model не умножаем на все оставшиеся ответы.
            progress.eta_seconds = answer_seconds / progress.completed * (
                progress.total - progress.completed
            ) + swap_seconds / max(1, loaded_checkpoints - 1) * (
                progress.total_checkpoints - loaded_checkpoints
            )
        if report_progress:
            report_progress(progress.model_copy(deep=True))
        terminal_log("CHECKPOINT EVAL", f"{stage}: {progress.completed}/{progress.total}")

    try:
        publish("Освобождение памяти")
        clear_pretrained_cache()
        config = GenerationConfig(
            checkpoint_file=checkpoints[0],
            tokenizer_file=run.tokenizer_file,
            pretrained_config_file=run.pretrained_config_file,
            model_backend="custom" if run.model_id == "custom" else "pretrained",
            pretrained_mode="custom_checkpoint",
            device=request.device,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )
        if not should_stop():
            session = session_factory(run, config)
        for path in checkpoints:
            if session is None or should_stop():
                progress.interrupted = True
                break
            progress.current_checkpoint = path.name
            progress.question_number = None
            progress.eta_seconds = None
            publish("Загрузка checkpoint")
            if signatures[path] != (path.stat().st_size, path.stat().st_mtime_ns):
                raise RuntimeError(f"Checkpoint изменился во время прогона: {path.name}")
            load_started = time.perf_counter()
            session.load(path)
            if loaded_checkpoints:
                swap_seconds += time.perf_counter() - load_started
            loaded_checkpoints += 1
            if signatures[path] != (path.stat().st_size, path.stat().st_mtime_ns):
                raise RuntimeError(f"Checkpoint изменился при загрузке: {path.name}")
            for index, question in enumerate(QUESTIONS):
                if should_stop():
                    progress.interrupted = True
                    break
                progress.question_number = index + 1
                publish("Генерация ответа")
                answer_started = time.perf_counter()
                devices = [torch.cuda.current_device()] if request.device == "cuda" else []
                # Один seed на номер вопроса для сравнения checkpoints; без памяти диалога.
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(request.seed + index)
                    result = session.generate(question)
                answer = EvaluationAnswer(
                    checkpoint=path.name,
                    question_number=index + 1,
                    question=question,
                    answer=result.text,
                    generated_tokens=len(result.token_ids),
                    elapsed_seconds=time.perf_counter() - answer_started,
                )
                answer_seconds += answer.elapsed_seconds
                append_answer(output, answer)
                progress.latest_result = answer
                progress.completed += 1
                publish("Ответ сохранён")
            if progress.interrupted:
                break
            progress.checkpoints_completed += 1
        message = "Остановлено пользователем" if progress.interrupted else "Завершено"
        finish_report(output, f"{message}. Ответов: {progress.completed}/{progress.total}.")
        publish(message)
        return {
            "report_file": output.as_posix(),
            "completed": progress.completed,
            "total": progress.total,
            "interrupted": progress.interrupted,
        }
    except Exception as error:
        finish_report(output, f"Ошибка: {type(error).__name__}: {error}")
        publish("Ошибка; частичный отчёт сохранён")
        raise
    finally:
        if session is not None:
            session.close()
