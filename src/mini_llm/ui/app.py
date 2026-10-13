"""Экран одного сообщения и ответа; история и память диалога не сохраняются."""

import streamlit as st

from mini_llm.ui.client import GenerationAPIError, generate_reply, get_generation_options
from mini_llm.ui.config import (
    MODE_LABELS,
    MODEL_BACKENDS,
    MODEL_LABELS,
    ModelBackend,
    PretrainedMode,
)


def main() -> None:
    """Показать минимальный чат и отправить текущий запрос в FastAPI."""

    st.set_page_config(page_title="НейроДвач", page_icon="💬", layout="centered")
    st.title("НейроДвач")
    backend: ModelBackend = st.selectbox(
        "Модель", MODEL_BACKENDS, format_func=lambda value: MODEL_LABELS[value]
    )
    mode: PretrainedMode | None = None
    pretrained_model_id: str | None = None
    checkpoint_name: str | None = None
    ready = True
    if backend == "pretrained":
        try:
            options = get_generation_options()
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
            if availability.after_sft_available:
                modes.append("after_sft")
            modes.append("custom_checkpoint")
            mode = st.radio(
                "Режим Qwen",
                modes,
                key=f"qwen_mode_{availability.model_id}",
                # При горячем обновлении Streamlit может сохранить старый словарь подписей.
                format_func=lambda value: MODE_LABELS.get(
                    value, "Свой чекпоинт" if value == "custom_checkpoint" else value
                ),
            )
            if mode == "custom_checkpoint":
                checkpoint_name = st.text_input(
                    "Имя чекпоинта",
                    key=f"qwen_checkpoint_{availability.model_id}",
                    placeholder="step_00000500.pt",
                    help="Только имя .pt файла. API выбирает каталог выбранной модели.",
                ).strip()
                ready = ready and bool(checkpoint_name)
            if availability.reason:
                st.caption(availability.reason + " Обновите страницу после подготовки/обучения.")
        except GenerationAPIError as error:
            st.error(str(error))
            ready = False
    prompt = st.chat_input("Напишите сообщение", disabled=not ready)
    if not prompt or not prompt.strip():
        return
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        with st.spinner("Модель готовит ответ…"):
            if pretrained_model_id is not None:
                reply = generate_reply(
                    prompt,
                    backend,
                    mode,
                    checkpoint_name=checkpoint_name,
                    pretrained_model_id=pretrained_model_id,
                )
            elif mode == "custom_checkpoint":
                reply = generate_reply(prompt, backend, mode, checkpoint_name=checkpoint_name)
            elif backend == "pretrained":
                reply = generate_reply(prompt, backend, mode)
            else:
                reply = generate_reply(prompt, backend)
    except GenerationAPIError as error:
        st.error(str(error))
        return
    with st.chat_message("assistant"):
        if reply:
            st.text(reply)
        else:
            st.caption("Модель вернула пустой ответ.")


if __name__ == "__main__":
    main()
