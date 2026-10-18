"""Живой прогресс обучения: шаги, метрики и приблизительный ETA."""

from datetime import UTC, datetime

import streamlit as st

from mini_llm.observability.terminal import format_elapsed
from mini_llm.observability.training import TrainingProgress


def render_training_progress(progress: TrainingProgress) -> None:
    """Отобразить снимок trainer без запуска вычислений или чтения checkpoint."""

    st.info(progress.stage)
    if progress.total_steps:
        percent = min(1.0, progress.global_step / progress.total_steps)
        st.progress(
            percent,
            text=(f"Шаг {progress.global_step} / {progress.total_steps} · {percent * 100:.1f}%"),
        )
    elapsed = progress.elapsed_seconds
    if progress.status in {"preparing", "running"}:
        elapsed = max(
            elapsed,
            (datetime.now(UTC) - datetime.fromisoformat(progress.started_at)).total_seconds(),
        )
    if progress.status == "completed":
        eta = "Готово"
    elif progress.status in {"failed", "interrupted"}:
        eta = "Остановлено"
    elif progress.total_steps and progress.global_step >= progress.total_steps:
        eta = "Финализация…"
    else:
        eta = (
            f"≈ {format_elapsed(progress.eta_seconds)}"
            if progress.eta_seconds is not None
            else "Оценивается…"
        )
    left, middle, right = st.columns(3)
    left.metric("Прошло", format_elapsed(elapsed))
    middle.metric("Осталось примерно", eta)
    right.metric(
        "Эпоха",
        (
            f"{progress.epoch} / {progress.planned_epochs} · {progress.epoch_progress:.1f}%"
            if progress.epoch and progress.planned_epochs
            else "Подготовка"
        ),
    )
    loss, rolling, validation = st.columns(3)
    loss.metric(
        "Train loss", f"{progress.train_loss:.4f}" if progress.train_loss is not None else "—"
    )
    rolling.metric(
        "Средний loss (100 шагов)",
        f"{progress.rolling_loss:.4f}" if progress.rolling_loss is not None else "—",
    )
    validation.metric(
        "Validation loss",
        f"{progress.validation_loss:.4f}" if progress.validation_loss is not None else "—",
    )
    speed = f"{progress.tokens_per_second:.1f}" if progress.tokens_per_second is not None else "—"
    st.caption(
        f"Samples: {progress.samples_seen} · Loss tokens: {progress.tokens_seen} · "
        f"Tokens/sec: {speed}"
    )
    if progress.start_step:
        st.caption(f"Resume с шага {progress.start_step}; ETA рассчитан по текущему запуску.")
    if progress.recent_losses:
        start = progress.global_step - len(progress.recent_losses) + 1
        st.line_chart(
            {"step": list(range(start, progress.global_step + 1)), "loss": progress.recent_losses},
            x="step",
            y="loss",
        )
    st.caption(
        "ETA приблизительный: уточняется по скорости, "
        "включая измеренные паузы на validation и checkpoints."
    )
