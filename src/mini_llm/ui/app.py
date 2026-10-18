"""Экран одного сообщения и ответа; история и память диалога не сохраняются."""

from importlib import reload

import streamlit as st

from mini_llm.ui import client
from mini_llm.ui.config import (
    MODE_LABELS,
    MODEL_BACKENDS,
    MODEL_LABELS,
    ModelBackend,
    PretrainedMode,
)

# Streamlit перезапускает скрипт, но сохраняет импортированные модули между обновлениями.
if getattr(client, "UI_CLIENT_VERSION", 0) < 2:
    reload(client)


def _select_experiment_checkpoint(
    experiments: list[client.ExperimentOptions], model_id: str
) -> tuple[str, str]:
    """Выбрать серверные имена вместо путей; свежий запуск и last.pt предлагаются первыми."""

    experiment_id: str = st.selectbox(
        "Версия модели / запуск",
        [item.experiment_id for item in experiments],
        format_func=lambda value: next(
            item.title or item.experiment_id for item in experiments if item.experiment_id == value
        ),
        key=f"experiment_{model_id}",
    )
    experiment = next(item for item in experiments if item.experiment_id == experiment_id)
    if not experiment.checkpoints:
        raise client.GenerationAPIError("В выбранном эксперименте нет доступных checkpoints")
    checkpoints = {item.name: item for item in experiment.checkpoints}
    names = list(checkpoints)
    checkpoint_name: str = st.selectbox(
        "Checkpoint",
        names,
        index=names.index("last.pt") if "last.pt" in names else 0,
        format_func=lambda name: (
            f"{name} · step {checkpoints[name].global_step}"
            if checkpoints[name].global_step is not None
            else f"{name} · шаг неизвестен"
        ),
        key=f"experiment_checkpoint_{model_id}_{experiment_id}",
    )
    directory = experiment.directory or f"checkpoints/runs/{experiment_id}"
    st.caption(f"Источник: {directory}/{checkpoint_name}")
    return experiment_id, checkpoint_name


def render_chat() -> None:
    """Показать минимальный чат и отправить текущий запрос в FastAPI."""

    st.title("НейроДвач")
    backend: ModelBackend = st.selectbox(
        "Модель", MODEL_BACKENDS, format_func=lambda value: MODEL_LABELS[value]
    )
    mode: PretrainedMode | None = None
    pretrained_model_id: str | None = None
    checkpoint_name: str | None = None
    experiment_id: str | None = None
    ready = True
    if backend == "pretrained":
        try:
            with st.spinner("Проверяем API и читаем каталог моделей/checkpoints…"):
                options = client.get_generation_options()
            if options.models:
                model_ids = [item.model_id for item in options.models]
                default_index = (
                    model_ids.index(options.model_id) if options.model_id in model_ids else 0
                )
                pretrained_model_id = st.selectbox("Модель Qwen", model_ids, index=default_index)
            availability = next(
                (item for item in options.models if item.model_id == pretrained_model_id), options
            )
            ready = availability.before_sft_available
            st.caption(availability.model_id)
            modes: list[PretrainedMode] = ["before_sft"]
            if availability.experiments:
                modes.insert(0, "custom_checkpoint")
            else:
                if availability.after_sft_available:
                    modes.append("after_sft")
                modes.append("custom_checkpoint")
            mode = st.radio(
                "Режим Qwen",
                modes,
                key=f"qwen_mode_{availability.model_id}"
                + ("_experiments" if availability.experiments else ""),
                # При горячем обновлении Streamlit может сохранить старый словарь подписей.
                format_func=lambda value: (
                    "После SFT — версия / запуск"
                    if availability.experiments and value == "custom_checkpoint"
                    else MODE_LABELS.get(
                        value, "Свой чекпоинт" if value == "custom_checkpoint" else value
                    )
                ),
            )
            if mode == "custom_checkpoint":
                if availability.experiments:
                    experiment_id, checkpoint_name = _select_experiment_checkpoint(
                        availability.experiments, availability.model_id
                    )
                else:
                    checkpoint_name = st.text_input(
                        "Имя чекпоинта",
                        key=f"qwen_checkpoint_{availability.model_id}",
                        placeholder="step_00000500.pt",
                        help="Только имя .pt файла. API выбирает каталог выбранной модели.",
                    ).strip()
                ready = ready and bool(checkpoint_name)
            if availability.reason and not availability.experiments:
                st.caption(availability.reason + " Обновите страницу после подготовки/обучения.")
            if not availability.experiments and availability.before_sft_available:
                st.caption(
                    "Совместимых запусков в checkpoints/runs нет. "
                    "Доступны прежние checkpoints зарегистрированной модели."
                )
        except client.GenerationAPIError as error:
            st.error(str(error))
            ready = False
    if backend == "custom":
        try:
            options = client.get_generation_options()
            if options.custom_runs:
                experiment_id, checkpoint_name = _select_experiment_checkpoint(
                    options.custom_runs, "custom"
                )
        except client.GenerationAPIError as error:
            st.warning(str(error))
    prompt = st.chat_input("Напишите сообщение", disabled=not ready)
    if not prompt or not prompt.strip():
        return
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        with st.spinner("Модель готовит ответ…"):
            if experiment_id is not None:
                reply = client.generate_reply(
                    prompt,
                    backend,
                    mode,
                    checkpoint_name=checkpoint_name,
                    pretrained_model_id=pretrained_model_id,
                    experiment_id=experiment_id,
                )
            elif pretrained_model_id is not None:
                reply = client.generate_reply(
                    prompt,
                    backend,
                    mode,
                    checkpoint_name=checkpoint_name,
                    pretrained_model_id=pretrained_model_id,
                )
            elif mode == "custom_checkpoint":
                reply = client.generate_reply(
                    prompt, backend, mode, checkpoint_name=checkpoint_name
                )
            elif backend == "pretrained":
                reply = client.generate_reply(prompt, backend, mode)
            else:
                reply = client.generate_reply(prompt, backend)
    except client.GenerationAPIError as error:
        st.error(str(error))
        return
    with st.chat_message("assistant"):
        if reply:
            st.text(reply)
        else:
            st.caption("Модель вернула пустой ответ.")


def main() -> None:
    """Переключать чат и обучение, не запуская pipeline при открытии страницы."""

    st.set_page_config(page_title="Mini LLM", page_icon="💬", layout="centered")
    page = st.segmented_control("Раздел", ["Чат", "Обучение"], default="Чат")
    if page == "Обучение":
        from mini_llm.ui.training import render_training

        render_training()
    else:
        render_chat()


if __name__ == "__main__":
    main()
