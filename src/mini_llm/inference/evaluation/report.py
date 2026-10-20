"""Единый Markdown-отчёт с немедленным сохранением каждого ответа."""

import json
import re
from pathlib import Path

from mini_llm.inference.evaluation.schemas import QUESTIONS, EvaluationAnswer, EvaluationRequest
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.inference import QWEN_SYSTEM_PROMPT
from mini_llm.workspace.catalog import ModelRun

REPORTS_ROOT = Path("artifacts/evaluations")


def literal_block(text: str) -> str:
    """Не позволять Markdown/HTML модели ломать границы вопросов и ответов."""

    fence = "`" * max(3, 1 + max((len(m[0]) for m in re.finditer(r"`+", text)), default=0))
    return f"{fence}text\n{text}\n{fence}\n"


def create_report(
    path: Path, request: EvaluationRequest, run: ModelRun, checkpoints: list[Path]
) -> None:
    """Создать новый файл исключительно, не заменяя предыдущие результаты."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": run.run_id,
        "model_id": run.model_id,
        "pretrained_config_file": str(run.pretrained_config_file),
        "tokenizer_file": str(run.tokenizer_file),
        "generation": request.model_dump(),
        "questions": list(QUESTIONS),
        "pretrained_config": PretrainedConfig.load(run.pretrained_config_file).model_dump(
            mode="json"
        )
        if run.pretrained_config_file
        else None,
        "checkpoints": [
            {"name": item.name, "size": item.stat().st_size, "mtime_ns": item.stat().st_mtime_ns}
            for item in checkpoints
        ],
        "system_prompt": QWEN_SYSTEM_PROMPT
        if run.model_id.rsplit("/", 1)[-1].lower().startswith("qwen")
        else None,
        "seed_policy": "Для каждого checkpoint: seed + номер вопроса (начиная с 0).",
    }
    with path.open("x", encoding="utf-8") as stream:
        stream.write("# Проверка checkpoints\n\n")
        stream.write(literal_block(json.dumps(metadata, ensure_ascii=False, indent=2)))


def append_answer(path: Path, answer: EvaluationAnswer) -> None:
    """Дописать и закрыть файл после каждого ответа, сохранив частичный прогон."""

    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n## {answer.checkpoint} · вопрос {answer.question_number}\n\n")
        stream.write(literal_block(answer.question))
        stream.write("\n### Ответ\n\n" + literal_block(answer.answer))
        stream.write(
            f"\nТокенов: {answer.generated_tokens}; секунд: {answer.elapsed_seconds:.2f}.\n"
        )


def finish_report(path: Path, message: str) -> None:
    """Зафиксировать завершение, остановку или ошибку без потери ответов."""

    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n## Итог\n\n" + literal_block(message))
