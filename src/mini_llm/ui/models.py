"""Загрузка HF-моделей и явная очистка выбранных локальных файлов."""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from mini_llm.observability import format_elapsed
from mini_llm.pretrained.management.schemas import (
    DeletionPreview,
    DeletionRequest,
    DeletionTarget,
    ManagedModel,
)
from mini_llm.ui import models_client
from mini_llm.ui.http import APIError


def format_size(size: int) -> str:
    """Отобразить приблизительный физический объём файлов."""

    for unit, factor in (("ГиБ", 1024**3), ("МиБ", 1024**2), ("КиБ", 1024)):
        if size >= factor:
            return f"{size / factor:.2f} {unit}"
    return f"{size} Б"


def _operation_status(job_id: str) -> None:
    try:
        job = models_client.get_model_job(job_id)
        if job.status == "failed":
            st.error(job.error or "Операция завершилась с ошибкой")
        elif job.status == "completed":
            message = (
                "Модель загружена и доступна в чате и обучении."
                if job.kind == "model_download"
                else "Выбранные файлы удалены, каталог обновлён."
            )
            st.success(message)
        else:
            elapsed = (
                datetime.now(UTC) - datetime.fromisoformat(job.started_at or job.created_at)
            ).total_seconds()
            stage = job.operation_progress.stage if job.operation_progress else "Ожидание запуска"
            st.info(f"{stage} · прошло {format_elapsed(elapsed)}")
            st.caption(
                "Подробности скачивания по файлам — в терминале API. Обучение не запускается."
                if job.kind == "model_download"
                else "Очищаются только подтверждённые пути."
            )
        if (
            job.status in {"completed", "failed"}
            and st.session_state.get("models_refreshed_job") != job_id
        ):
            st.session_state["models_refreshed_job"] = job_id
            st.rerun()
    except APIError as error:
        st.error(str(error))


@st.fragment(run_every=2.0)
def _watch_operation(job_id: str) -> None:
    """Опросить только явно запущенную загрузку/очистку, не весь каталог."""

    _operation_status(job_id)


def _delete_target(model: ManagedModel) -> DeletionTarget | None:
    scope = st.radio(
        "Что удалить",
        ["checkpoint", "run", "model"],
        index=1 if model.runs else 2,
        format_func=lambda value: {
            "checkpoint": "Один checkpoint",
            "run": "Один запуск целиком",
            "model": "Вся модель и её запуски",
        }[value],
        key=f"delete_scope_{model.model_id}",
    )
    if scope == "model":
        include_cache = (
            False
            if model.model_id == "custom"
            else st.checkbox(
                "Также удалить скачанные базовые веса и tokenizer из HF cache",
                value=True,
                key=f"delete_cache_{model.model_id}",
            )
        )
        if model.model_id == "custom":
            st.caption(
                "У Custom удаляются только запуски/checkpoints. Код и общий BPE сохраняются."
            )
        return DeletionTarget(scope="model", model_id=model.model_id, include_cache=include_cache)
    if not model.runs:
        st.info("У этой модели пока нет запусков.")
        return None
    runs = {item.run_id: item for item in model.runs}
    run_id = st.selectbox(
        "Запуск для очистки",
        list(runs),
        format_func=lambda value: runs[value].title,
        key=f"delete_run_{model.model_id}",
    )
    if scope == "run":
        return DeletionTarget(scope="run", run_id=run_id)
    if not runs[run_id].checkpoints:
        st.info("В этом запуске нет checkpoints.")
        return None
    name = st.selectbox(
        "Checkpoint для удаления", runs[run_id].checkpoints, key=f"delete_checkpoint_{run_id}"
    )
    return DeletionTarget(scope="checkpoint", run_id=run_id, checkpoint_name=name)


def _render_deletion(model: ManagedModel, busy: bool) -> None:
    target = _delete_target(model)
    if target is None:
        return
    if st.button("Показать план удаления", disabled=busy):
        st.session_state.pop("model_delete_preview", None)
        try:
            st.session_state["model_delete_preview"] = models_client.preview_deletion(
                target
            ).model_dump()
        except APIError as error:
            st.error(str(error))
    saved = st.session_state.get("model_delete_preview")
    if not isinstance(saved, dict):
        return
    preview = DeletionPreview.model_validate(saved)
    if preview.target != target:
        return
    st.warning(preview.warning)
    st.write(f"Будет удалено: {preview.files} файлов, примерно {format_size(preview.size_bytes)}")
    for path in preview.paths:
        st.code(f"{path.path}  ({format_size(path.size_bytes)})", language=None)
    confirmation = st.text_input(
        "Для подтверждения введите УДАЛИТЬ", key=f"confirm_{preview.preview_token}"
    )
    if st.button(
        "Удалить безвозвратно", type="primary", disabled=busy or confirmation != "УДАЛИТЬ"
    ):
        try:
            job = models_client.delete_files(
                DeletionRequest(
                    target=target, preview_token=preview.preview_token, confirmation="УДАЛИТЬ"
                )
            )
            st.session_state["models_job_id"] = job.job_id
            st.session_state.pop("model_delete_preview", None)
            st.rerun()
        except APIError as error:
            st.error(str(error))


def render_models() -> None:
    """Обновить каталог по открытию/кнопке; действия выполнять только по явному подтверждению."""

    st.title("Модели")
    st.button("Обновить каталог моделей", key="refresh_models")
    try:
        catalog = models_client.get_models()
    except APIError as error:
        st.error(str(error))
        return
    busy = catalog.active_job_id is not None
    if busy:
        st.warning(
            "Выполняется фоновая задача. Загрузка и удаление заблокированы до её завершения."
        )
    job_id = st.session_state.get("models_job_id")
    if isinstance(job_id, str):
        if st.session_state.get("models_refreshed_job") == job_id:
            _operation_status(job_id)
        else:
            _watch_operation(job_id)
    st.subheader("Загрузить с Hugging Face")
    with st.form("download_hf_model"):
        model_id = st.text_input("Hugging Face model ID", placeholder="Qwen/Qwen3-0.6B")
        download = st.form_submit_button("Загрузить модель", disabled=busy)
    st.caption(
        "Нужен ID decoder-only модели. Сервер скачает файлы, закрепит commit и создаст "
        "QLoRA-конфиг. GPU и обучение при загрузке не используются; "
        "gated-модели требуют настроенной авторизации HF."
    )
    if download:
        if not model_id.strip():
            st.error("Укажите Hugging Face model ID")
        else:
            try:
                job = models_client.download_model(model_id.strip())
                st.session_state["models_job_id"] = job.job_id
                st.rerun()
            except APIError as error:
                st.error(str(error))
    st.subheader("Локальные модели и очистка")
    st.caption(
        f"Моделей: {len(catalog.models)} · Файлы проекта: ≈ {format_size(catalog.total_size_bytes)}"
    )
    for warning in catalog.warnings:
        st.warning(warning)
    if not catalog.models:
        st.info("Локальных моделей пока нет.")
        return
    models = {item.model_id: item for item in catalog.models}
    model_ids = list(models)
    selected = st.selectbox(
        "Локальная модель",
        model_ids,
        index=next(
            (index for index, model_id in enumerate(model_ids) if models[model_id].active), 0
        ),
        format_func=lambda value: (
            f"{'Custom' if value == 'custom' else value} · "
            f"{format_size(models[value].size_bytes)} · запусков {len(models[value].runs)}"
        ),
    )
    model = models[selected]
    if model.active:
        st.caption("Активная pretrained-модель по умолчанию")
    if model.model_id == "custom":
        st.write("Собственная архитектура. Для генерации нужен обученный запуск.")
    elif not model.registered:
        st.info(
            "Незарегистрированные артефакты. "
            "Загрузите модель заново для использования или удалите их."
        )
    else:
        st.write(
            "Локальные файлы доступны"
            if model.available
            else "Файлы не найдены; модель можно загрузить заново"
        )
    if model.revision:
        st.caption(f"Revision: {model.revision}")
    if model.config_file:
        st.caption(f"Конфиг: {model.config_file}")
    if model.snapshot_path:
        st.caption(f"Файлы: {model.snapshot_path}")
    st.caption(
        "Объём относится к диску, не VRAM. Не запускайте внешние CLI-процессы "
        "с этими файлами во время очистки."
    )
    _render_deletion(model, busy)
