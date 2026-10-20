"""Минимальный экран проверки всех checkpoints выбранного запуска."""

from datetime import UTC, datetime

import streamlit as st

from mini_llm.ui import evaluation_client, training_client
from mini_llm.ui.http import APIError


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "Оцениваем…"
    minutes, remainder = divmod(max(0, int(seconds)), 60)
    return f"{minutes} мин {remainder:02d} с"


@st.fragment(run_every=2.0)
def _show_evaluation(job_id: str) -> None:
    """Обновлять только текущий прогресс; ответы не добавляются в историю чата."""

    try:
        job = evaluation_client.get_job(job_id)
        if job.kind != "checkpoint_evaluation":
            st.info("GPU занят другой задачей")
            return
        snapshot = job.evaluation_progress
        if snapshot:
            elapsed = snapshot.elapsed_seconds
            if job.started_at and job.status == "running":
                elapsed = (
                    datetime.now(UTC) - datetime.fromisoformat(job.started_at)
                ).total_seconds()
            st.progress(
                min(1.0, snapshot.completed / max(snapshot.total, 1)),
                text=(
                    f"Готово {snapshot.completed} из {snapshot.total} · "
                    f"осталось {snapshot.total - snapshot.completed}"
                ),
            )
            left, right = st.columns(2)
            left.metric(
                "Checkpoints", f"{snapshot.checkpoints_completed} / {snapshot.total_checkpoints}"
            )
            right.metric("Примерно осталось", _duration(snapshot.eta_seconds))
            st.caption(
                f"{snapshot.stage} · {snapshot.current_checkpoint or '—'} · "
                f"прошло {_duration(elapsed)}"
            )
            if snapshot.latest_result:
                result = snapshot.latest_result
                st.caption(
                    f"Последний ответ · {result.checkpoint} · вопрос {result.question_number}"
                )
                with st.chat_message("user"):
                    st.text(result.question)
                with st.chat_message("assistant"):
                    st.text(result.answer or "Пустой ответ")
            st.caption(snapshot.report_file)
            st.link_button(
                "Скачать Markdown",
                evaluation_client.report_url(job_id),
                disabled=job.status in {"queued", "running"},
            )
        else:
            st.info("Подготовка прогона…")
        if job.status in {"queued", "running"}:
            if st.button("Остановить", key=f"stop_eval_{job_id}", disabled=job.stop_requested):
                evaluation_client.stop_evaluation(job_id)
                st.rerun()
            if job.stop_requested:
                st.caption("Сохраним текущий ответ и остановимся")
        elif job.status == "failed":
            st.error(job.error or "Ошибка прогона; готовые ответы сохранены")
        elif snapshot and snapshot.interrupted:
            st.warning("Остановлено. Частичный отчёт сохранён")
        elif job.status == "completed":
            st.success("Проверка завершена")
    except APIError as error:
        st.error(str(error))


def render_evaluation() -> None:
    """Выбрать модель и запуск, затем явно поставить проверку в общую очередь API."""

    st.title("Проверка checkpoints")
    try:
        workspace = training_client.get_workspace(refresh_datasets=False)
        runs = [run for run in workspace.runs if run.checkpoints]
        if runs:
            models = list(dict.fromkeys(run.model_id for run in runs))
            model = st.selectbox("Модель", models, key="evaluation_model")
            versions = {run.run_id: run for run in runs if run.model_id == model}
            run_id = st.selectbox(
                "Запуск",
                list(versions),
                format_func=lambda key: versions[key].title,
                key="evaluation_run",
            )
            plan = evaluation_client.get_plan(run_id)
            st.caption(
                f"{len(plan.checkpoints)} checkpoints × {len(plan.questions)} вопросов "
                f"= {plan.total} ответов · до 1024 новых токенов"
            )
            with st.expander("Вопросы и checkpoints"):
                for question in plan.questions:
                    st.text(question)
                st.text(", ".join(plan.checkpoints))
            if st.button(
                "Проверить все checkpoints",
                type="primary",
                disabled=workspace.active_job_id is not None,
            ):
                job = evaluation_client.start_evaluation(run_id)
                st.session_state["evaluation_job_id"] = job.job_id
        else:
            st.info("Нет запусков с checkpoints")
        job_id = st.session_state.get("evaluation_job_id", workspace.active_job_id)
        if isinstance(job_id, str):
            _show_evaluation(job_id)
    except APIError as error:
        st.error(str(error))
