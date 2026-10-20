"""Минимальный чат без истории и общий экран локального проекта."""

import streamlit as st

from mini_llm.ui import client
from mini_llm.ui.config import ModelBackend, PretrainedMode


def render_chat() -> None:
    """Выбрать модель, версию и checkpoint, не вводя серверные пути."""

    st.title("НейроДвач")
    try:
        options = client.get_generation_options()
    except client.GenerationAPIError as error:
        st.error(str(error))
        return
    models = {item.model_id: item for item in options.models}
    if not models and options.model_id:
        models[options.model_id] = options
    model_id = st.selectbox(
        "Модель",
        ["custom", *models],
        format_func=lambda value: "Custom" if value == "custom" else value,
    )
    backend: ModelBackend = "custom" if model_id == "custom" else "pretrained"
    availability = models.get(model_id)
    experiments = (
        options.custom_runs
        if backend == "custom"
        else (availability.experiments if availability else [])
    )
    runs = {item.experiment_id: item for item in experiments}
    versions = list(runs)
    if backend == "pretrained" and availability and availability.before_sft_available:
        versions.append("")
    mode: PretrainedMode | None = "before_sft" if backend == "pretrained" else None
    experiment_id = checkpoint_name = None
    ready = bool(versions)
    if versions:
        version = st.selectbox(
            "Версия",
            versions,
            format_func=lambda value: (runs[value].title or value) if value else "Базовая · до SFT",
            key=f"version_{model_id}",
        )
        if version:
            experiment_id = version
            names = [item.name for item in runs[version].checkpoints]
            checkpoint_name = (
                st.selectbox(
                    "Checkpoint",
                    names,
                    index=names.index("last.pt") if "last.pt" in names else 0,
                    key=f"checkpoint_{model_id}_{version}",
                )
                if names
                else None
            )
            mode = "custom_checkpoint" if backend == "pretrained" else None
            ready = checkpoint_name is not None
    else:
        st.info("Нет доступных весов. Загрузите модель или завершите обучение.")
    prompt = st.chat_input("Напишите сообщение", disabled=not ready)
    if not prompt or not prompt.strip():
        return
    with st.chat_message("user"):
        st.text(prompt)
    with st.chat_message("assistant"):
        output = st.empty()
        reply = ""
        try:
            with st.spinner("Генерация…"):
                for reply in client.stream_reply(
                    prompt,
                    backend,
                    mode,
                    checkpoint_name=checkpoint_name,
                    pretrained_model_id=model_id if backend == "pretrained" else None,
                    experiment_id=experiment_id,
                ):
                    # Снимки позволяют корректно дополнять BPE и показывать текст без Markdown.
                    output.text(reply)
            output.text(reply or "Пустой ответ")
        except client.GenerationAPIError as error:
            st.error(str(error))


def main() -> None:
    """Открытие раздела не запускает фоновые операции."""

    st.set_page_config(page_title="НейроДвач", page_icon="💬", layout="centered")
    st.html("""
    <style>
    .block-container {max-width: 860px; padding-top: 2rem;}
    [data-testid="stChatMessage"] {border-radius: 18px; padding: 1.2rem;}
    [data-testid="stSegmentedControl"] {margin-bottom: 1.5rem;}
    h1 {letter-spacing: -.04em;}
    </style>
    """)
    page = st.segmented_control(
        "Раздел", ["Чат", "Обучение", "Модели", "Проверка checkpoints"], default="Чат"
    )
    if page == "Обучение":
        from mini_llm.ui.training import render_training

        render_training()
    elif page == "Модели":
        from mini_llm.ui.models import render_models

        render_models()
    elif page == "Проверка checkpoints":
        from mini_llm.ui.evaluation import render_evaluation

        render_evaluation()
    else:
        render_chat()


if __name__ == "__main__":
    main()
