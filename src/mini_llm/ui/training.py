"""Форма управляемого обучения и мониторинг фоновой задачи API."""

from __future__ import annotations

import streamlit as st

from mini_llm.data.config import MAX_BATCH_SIZE
from mini_llm.observability.training import TrainingProgress
from mini_llm.ui import training_client
from mini_llm.ui.http import APIError as GenerationAPIError
from mini_llm.ui.progress import render_training_progress


@st.fragment(run_every=2.0)
def _show_job(job_id: str) -> None:
    """Обновлять только статус и progress; не отправлять POST повторно."""

    try:
        job = training_client.get_job(job_id)
        progress = getattr(job, "progress", None)
        live_progress = progress if isinstance(progress, TrainingProgress) else None
        st.write(f"Задача: {job.status}")
        options = training_client.get_workspace(refresh_datasets=False)
        run = next((item for item in options.runs if item.job_id == job_id), None)
        if run:
            st.caption(run.title)
            st.caption(f"Сохранение: {run.directory}")
            if run.planned_total_steps and live_progress is None:
                st.progress(
                    min(1.0, run.global_step / run.planned_total_steps),
                    text=f"Сохранённый шаг {run.global_step} / {run.planned_total_steps}",
                )
            if run.status == "failed":
                st.error(run.error or "Ошибка обучения; проверьте терминал backend")
        if live_progress is not None:
            render_training_progress(live_progress)
        elif job.status in {"queued", "running"}:
            st.info("Подготовка запуска; ожидаем первые метрики trainer…")
        if job.status == "failed":
            st.error(job.error or "Задача завершилась с ошибкой")
        elif run and run.status == "interrupted":
            st.warning("Обучение прервано; последний checkpoint сохранён. Можно продолжить запуск.")
        elif job.status == "completed":
            st.success("Готово. Обновите каталог в чате для выбора новой версии модели.")
    except GenerationAPIError as error:
        st.error(str(error))


def render_training() -> None:
    """Показать параметры; старт разрешён только по кнопке отправки формы."""

    st.title("Обучение модели")
    # Кнопка вызывает обычный rerun: ниже выполняется один свежий запрос без кэша.
    st.button("Обновить список датасетов", key="refresh_datasets")
    try:
        options = training_client.get_workspace()
    except GenerationAPIError as error:
        st.error(str(error))
        return
    count = options.dataset_count if options.dataset_count is not None else len(options.datasets)
    total = (
        options.total_samples
        if options.total_samples is not None
        else sum(item.samples for item in options.datasets)
    )
    st.caption(f"Версий датасетов: {count} · Всего samples: {total}")
    for warning in options.dataset_warnings:
        st.warning(warning)
    if not options.datasets or not options.models:
        st.info("Сначала опубликуйте версию датасета и подготовьте модель через pretrained setup.")
        return
    datasets = {f"{item.dataset_id}/{item.version}": item for item in reversed(options.datasets)}
    dataset_key = st.selectbox(
        "Версия датасета",
        list(datasets),
        format_func=lambda key: f"{key} — {datasets[key].title} ({datasets[key].samples} samples)",
    )
    dataset = datasets[dataset_key]
    models = {item.model_id: item for item in options.models}
    model_id = st.selectbox("Модель для обучения", list(models), index=len(models) - 1)
    model = models[model_id]
    st.caption(
        f"Режим: {model.adaptation_mode}. Новые checkpoints сохраняются в отдельный run-NNN."
    )
    if not dataset.splits_ready:
        st.warning("У этой версии ещё нет train/validation/test splits.")
        if st.button("Подготовить splits", disabled=options.active_job_id is not None):
            try:
                st.session_state["training_job_id"] = training_client.prepare_splits(
                    dataset.dataset_id, dataset.version
                ).job_id
            except GenerationAPIError as error:
                st.error(str(error))
        if "training_job_id" in st.session_state:
            _show_job(str(st.session_state["training_job_id"]))
        return
    parents = {
        item.run_id: item
        for item in options.runs
        if item.model_id == model_id
        and item.dataset_id == dataset.dataset_id
        and item.dataset_version == dataset.version
    }
    parent_id = st.selectbox(
        "Продолжить запуск",
        ["", *parents],
        format_func=lambda key: parents[key].title if key else "Новый независимый запуск",
    )
    parent = parents.get(parent_id)
    saved = parent.training_config if parent else {}
    is_custom = model_id == "custom"
    prefix = f"train_{dataset_key}_{model_id}_{parent_id}"
    if parent:
        st.info(
            f"Resume из last.pt. Общий план включает выполненные {parent.global_step} шагов. "
            "LR можно изменить явно; результат сохранится в новом запуске."
        )
    with st.form(prefix):
        left, right = st.columns(2)
        with left:
            batch = st.number_input(
                "Batch size",
                1,
                MAX_BATCH_SIZE,
                int(str(saved.get("batch_size", 4 if is_custom else 2))),
            )
            lr = st.number_input(
                "Learning rate",
                min_value=0.0000001,
                value=float(str(saved.get("learning_rate", 0.0003 if is_custom else 0.00003))),
                format="%.7f",
            )
            duration = st.selectbox("Длительность", ["Epochs", "Max steps"])
            amount = st.number_input(
                "Общее количество epochs / steps",
                min_value=1,
                value=(int(str(saved.get("epochs") or 1)) + 1) if parent else 1,
            )
            seq = st.number_input(
                "Sequence length", min_value=2, value=model.max_sequence_length, disabled=is_custom
            )
            workers = st.number_input(
                "DataLoader workers",
                min_value=0,
                value=int(str(saved.get("num_workers", 0 if is_custom else 2))),
            )
            seed = st.number_input("Seed", value=int(str(saved.get("random_seed", 42))))
            limit = st.number_input(
                "Максимум train samples (0 = все)",
                min_value=0,
                value=int(str(saved.get("max_train_samples") or 0)),
            )
            validation_limit = st.number_input(
                "Максимум validation samples (0 = все)",
                min_value=0,
                value=int(str(saved.get("max_validation_samples") or 0)),
            )
        with right:
            val_interval = st.number_input("Validation interval", min_value=1, value=100)
            val_batches = st.number_input("Validation batches (0 = все)", min_value=0, value=50)
            checkpoint_interval = st.number_input("Checkpoint interval", min_value=1, value=100)
            log_interval = st.number_input("Log interval", min_value=1, value=25)
            decay = st.number_input("Weight decay", min_value=0.0, value=0.01)
            clip = st.number_input("Gradient clip norm", min_value=0.01, value=1.0)
            amp = st.checkbox("Mixed precision", value=True)
            gradient_checkpointing = st.checkbox(
                "Gradient checkpointing", value=False, disabled=is_custom
            )
            device = st.selectbox("Device", ["cuda", "cpu"])
        submitted = st.form_submit_button(
            "Запустить обучение", disabled=options.active_job_id is not None
        )
    if submitted:
        body: dict[str, object] = {
            "dataset_id": dataset.dataset_id,
            "dataset_version": dataset.version,
            "model_id": model_id,
            "parent_run_id": parent_id or None,
            "parameters": {
                "batch_size": batch,
                "learning_rate": lr,
                "epochs": amount if duration == "Epochs" else None,
                "max_steps": amount if duration == "Max steps" else None,
                "max_sequence_length": seq,
                "num_workers": workers,
                "random_seed": seed,
                "max_train_samples": limit or None,
                "max_validation_samples": validation_limit or None,
                "validation_interval": val_interval,
                "validation_batches": val_batches,
                "checkpoint_interval": checkpoint_interval,
                "log_interval": log_interval,
                "weight_decay": decay,
                "gradient_clip_norm": clip,
                "mixed_precision": amp,
                "gradient_checkpointing": gradient_checkpointing,
                "device": device,
            },
        }
        try:
            job = training_client.start_training(body)
            st.session_state["training_job_id"] = job.job_id
        except GenerationAPIError as error:
            st.error(str(error))
    job_id = st.session_state.get("training_job_id", options.active_job_id)
    if job_id:
        _show_job(str(job_id))
    st.caption(
        "Обучение выполняет backend. Закрытие браузера не останавливает задачу; "
        "подробный лог — в терминале API."
    )
