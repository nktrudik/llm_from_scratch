"""Форма обучения, прогресс и кооперативная остановка через API."""

from typing import Literal

import streamlit as st

from mini_llm.data.config import MAX_BATCH_SIZE
from mini_llm.ui import training_client
from mini_llm.ui.http import APIError
from mini_llm.ui.progress import render_training_progress
from mini_llm.workspace.training import TrainingParameters


@st.fragment(run_every=2.0)
def _show_job(job_id: str) -> None:
    """Опросить метрики, не повторяя запуск или сканирование датасетов."""

    try:
        job = training_client.get_job(job_id)
        if job.progress is not None:
            render_training_progress(job.progress)
        elif job.status in {"queued", "running"}:
            st.info("Подготовка запуска…")
        if job.status in {"queued", "running"} and job.kind == "training":
            if st.button("Остановить обучение", disabled=job.stop_requested, key=f"stop_{job_id}"):
                training_client.stop_training(job_id)
                st.rerun()
            if job.stop_requested:
                st.caption("Завершаем текущий шаг и сохраняем checkpoint…")
        elif job.status == "failed":
            st.error(job.error or "Ошибка обучения")
        elif job.progress and job.progress.status == "interrupted":
            st.warning("Обучение остановлено")
        elif job.status == "completed":
            st.success("Запуск завершён")
            if job.kind == "dataset_split" and st.session_state.get("split_refreshed") != job_id:
                st.session_state["split_refreshed"] = job_id
                st.rerun()
    except APIError as error:
        st.error(str(error))


def _parameters(saved: dict[str, object], *, custom: bool, seq: int) -> TrainingParameters:
    """Основные параметры видимы сразу; редкие настройки сгруппированы отдельно."""

    left, right = st.columns(2)
    with left:
        batch = st.number_input(
            "Batch size", 1, MAX_BATCH_SIZE, int(str(saved.get("batch_size", 4 if custom else 2)))
        )
        lr = st.number_input(
            "Learning rate",
            min_value=0.0000001,
            value=float(str(saved.get("learning_rate", 0.0003 if custom else 0.00003))),
            format="%.7f",
        )
    with right:
        sequence = st.number_input("Sequence length", min_value=2, value=seq, disabled=custom)
        epochs = st.number_input(
            "Всего эпох", min_value=1, value=int(str(saved.get("epochs") or 1))
        )
    with st.expander("Настройки"):
        engine: Literal["transformers", "unsloth"] = st.selectbox(
            "Движок SFT",
            ["transformers", "unsloth"],
            index=1 if saved.get("pretrained_engine") == "unsloth" else 0,
            disabled=custom,
        )
        if engine == "unsloth":
            st.caption("Нужен extra unsloth. Перезапустите API до загрузки/генерации модели.")
        left, right = st.columns(2)
        with left:
            rank = st.number_input(
                "LoRA rank", min_value=1, value=int(str(saved.get("lora_rank", 8))), disabled=custom
            )
            alpha = st.number_input(
                "LoRA alpha",
                min_value=1,
                value=int(str(saved.get("lora_alpha", 16))),
                disabled=custom,
            )
            workers = st.number_input(
                "DataLoader workers",
                min_value=0,
                value=int(str(saved.get("num_workers", 0 if custom else 2))),
            )
            seed = st.number_input("Seed", value=int(str(saved.get("random_seed", 42))))
            limit = st.number_input(
                "Максимум train samples (0 = все)",
                min_value=0,
                value=int(str(saved.get("max_train_samples") or 0)),
            )
            val_limit = st.number_input(
                "Максимум validation samples (0 = все)",
                min_value=0,
                value=int(str(saved.get("max_validation_samples") or 0)),
            )
            max_steps = st.number_input(
                "Max steps (0 = использовать эпохи)",
                min_value=0,
                value=int(str(saved.get("max_steps") or 0)),
            )
        with right:
            val_interval = st.number_input(
                "Validation interval",
                min_value=1,
                value=int(str(saved.get("validation_interval", 100))),
            )
            val_batches = st.number_input(
                "Validation batches (0 = все)",
                min_value=0,
                value=int(str(saved.get("validation_batches", 50))),
            )
            checkpoint_interval = st.number_input(
                "Checkpoint interval",
                min_value=1,
                value=int(str(saved.get("checkpoint_interval", 100))),
            )
            log_interval = st.number_input(
                "Log interval", min_value=1, value=int(str(saved.get("log_interval", 25)))
            )
            decay = st.number_input(
                "Weight decay", min_value=0.0, value=float(str(saved.get("weight_decay", 0.01)))
            )
            clip = st.number_input(
                "Gradient clip norm",
                min_value=0.01,
                value=float(str(saved.get("gradient_clip_norm", 1.0))),
            )
            amp = st.checkbox("Mixed precision", value=bool(saved.get("mixed_precision", True)))
            checkpointing = st.checkbox(
                "Gradient checkpointing",
                value=bool(saved.get("gradient_checkpointing", False)),
                disabled=custom,
            )
            device: Literal["cuda", "cpu"] = st.selectbox("Device", ["cuda", "cpu"])
    return TrainingParameters(
        batch_size=batch,
        learning_rate=lr,
        max_sequence_length=sequence,
        epochs=None if max_steps else epochs,
        max_steps=max_steps or None,
        pretrained_engine=engine,
        lora_rank=rank,
        lora_alpha=alpha,
        num_workers=workers,
        random_seed=seed,
        max_train_samples=limit or None,
        max_validation_samples=val_limit or None,
        validation_interval=val_interval,
        validation_batches=val_batches,
        checkpoint_interval=checkpoint_interval,
        log_interval=log_interval,
        weight_decay=decay,
        gradient_clip_norm=clip,
        mixed_precision=amp,
        gradient_checkpointing=checkpointing,
        device=device,
    )


def render_training() -> None:
    """Новый запуск или resume всегда сохраняется в отдельный run-NNN."""

    st.title("Обучение")
    st.button("Обновить список датасетов", key="refresh_datasets", icon=":material/refresh:")
    try:
        options = training_client.get_workspace()
    except APIError as error:
        st.error(str(error))
        return
    for warning in options.dataset_warnings:
        st.warning(warning)
    if not options.datasets or not options.models:
        st.info("Добавьте датасет и загрузите модель в разделе «Модели».")
        return
    datasets = {f"{item.dataset_id}/{item.version}": item for item in reversed(options.datasets)}
    dataset_key = st.selectbox(
        "Версия датасета",
        list(datasets),
        format_func=lambda key: f"{key} · {datasets[key].title} · {datasets[key].samples}",
    )
    dataset = datasets[dataset_key]
    models = {item.model_id: item for item in options.models}
    model_id = st.selectbox("Модель для обучения", list(models), index=len(models) - 1)
    if not dataset.splits_ready:
        st.info("Для этой версии ещё не подготовлены splits.")
        if st.button("Подготовить splits", disabled=options.active_job_id is not None):
            try:
                st.session_state["training_job_id"] = training_client.prepare_splits(
                    dataset.dataset_id, dataset.version
                ).job_id
            except APIError as error:
                st.error(str(error))
        if "training_job_id" in st.session_state:
            _show_job(str(st.session_state["training_job_id"]))
        return
    parents = {
        item.run_id: item
        for item in options.runs
        if (
            item.model_id == model_id
            and item.dataset_id == dataset.dataset_id
            and item.dataset_version == dataset.version
            and (model_id == "custom" or item.training_config.get("trainer") == "trl")
            and item.checkpoints
        )
    }
    parent_id = st.selectbox(
        "Продолжить запуск",
        ["", *parents],
        format_func=lambda key: parents[key].title if key else "Новый запуск",
    )
    parent = parents.get(parent_id)
    saved = parent.training_config if parent else {}
    checkpoint = "last.pt"
    if parent:
        names = list(parent.checkpoints)
        checkpoint = st.selectbox(
            "Продолжить из checkpoint",
            names,
            index=names.index("last.pt") if "last.pt" in names else 0,
        )
    prefix = f"train_{dataset_key}_{model_id}_{parent_id}"
    with st.form(prefix):
        parameters = _parameters(
            saved,
            custom=model_id == "custom",
            seq=int(str(saved.get("max_sequence_length", models[model_id].max_sequence_length))),
        )
        submitted = st.form_submit_button(
            "Запустить обучение", type="primary", disabled=options.active_job_id is not None
        )
    if submitted:
        try:
            job = training_client.start_training(
                {
                    "dataset_id": dataset.dataset_id,
                    "dataset_version": dataset.version,
                    "model_id": model_id,
                    "parent_run_id": parent_id or None,
                    "checkpoint_name": checkpoint,
                    "parameters": parameters.model_dump(),
                }
            )
            st.session_state["training_job_id"] = job.job_id
        except APIError as error:
            st.error(str(error))
    job_id = st.session_state.get("training_job_id", options.active_job_id)
    if job_id:
        _show_job(str(job_id))
