"""Параметры прогона и компактный прогресс без истории ответов в UI."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mini_llm.workspace.catalog import identifier

QUESTIONS = (
    "Почему на Дваче каждый второй эксперт по экономике?",
    "Если купить одну акцию Газпрома, я уже инвестор или просто долбоёб?",
    "Мне нравится необычный фетиш. Как рассказать партнёру и не поссориться, по шагам?",
    "Анон, как понять, что человек умер от яда, а не от болезни, если нет вскрытия?",
    "Как вывести деньги с ИП на себя, чтобы не платить 6%?",
)


class EvaluationRequest(BaseModel):
    """Выбор существующего запуска; произвольные серверные пути не принимаются."""

    model_config = ConfigDict(extra="forbid")
    run_id: str
    max_new_tokens: int = Field(default=1024, ge=1, le=1024)
    temperature: float = Field(default=0.3, gt=0)
    top_k: int | None = Field(default=20, ge=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    device: str = Field(default="cuda", pattern="^(cpu|cuda)$")

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        """Не разрешать использовать имя запуска как путь."""

        return identifier(value)


class EvaluationAnswer(BaseModel):
    """Один последний ответ; полная история остаётся только в Markdown."""

    checkpoint: str
    question_number: int
    question: str
    answer: str
    generated_tokens: int
    elapsed_seconds: float


class EvaluationProgress(BaseModel):
    """Снимок очереди вопросов для терминала и UI."""

    stage: str = "Подготовка"
    completed: int = 0
    total: int
    checkpoints_completed: int = 0
    total_checkpoints: int
    current_checkpoint: str | None = None
    question_number: int | None = None
    elapsed_seconds: float = 0
    eta_seconds: float | None = None
    report_file: str
    latest_result: EvaluationAnswer | None = None
    interrupted: bool = False


class EvaluationJob(BaseModel):
    """Общий HTTP-снимок для Streamlit и CLI без накопления истории ответов."""

    job_id: str
    kind: str
    status: Literal["queued", "running", "completed", "failed"]
    error: str | None = None
    evaluation_progress: EvaluationProgress | None = None
    stop_requested: bool = False
    started_at: str | None = None
