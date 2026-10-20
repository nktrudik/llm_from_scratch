"""Потоковая генерация с единым GPU-слотом и остановкой при отключении клиента."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from queue import Empty, Full, Queue
from threading import Event, Thread

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from mini_llm.api.jobs import JobManager
from mini_llm.api.schemas import GenerationRequest
from mini_llm.inference.service import generate_response


class _StreamClosed(RuntimeError):
    """Клиент отключился; завершить вычисления после текущего токена."""


def generation_stream(request: GenerationRequest, manager: JobManager) -> StreamingResponse:
    """Передавать NDJSON-снимки текста; обычная ручка генерации остаётся совместимой."""

    if manager.active_job_id() is not None:
        raise HTTPException(
            status_code=409, detail="Генерация недоступна: выполняется фоновая задача"
        )
    config = request.to_config()

    async def events() -> AsyncIterator[str]:
        queue: Queue[dict[str, object]] = Queue(maxsize=32)
        closed, finished = Event(), Event()

        def emit(payload: dict[str, object]) -> None:
            while not closed.is_set():
                try:
                    queue.put(payload, timeout=0.1)
                    return
                except Full:
                    continue
            raise _StreamClosed

        def work() -> None:
            previous = ""

            def on_text(text: str) -> None:
                nonlocal previous
                if closed.is_set():
                    raise _StreamClosed
                if text != previous:
                    emit({"type": "text", "text": text})
                    previous = text

            try:
                with manager.generation():
                    if closed.is_set():
                        return
                    result = generate_response(request.prompt, config, on_text=on_text)
                    emit({"type": "done", "text": result.text})
            except _StreamClosed:
                pass
            except Exception as error:
                with suppress(_StreamClosed):
                    emit({"type": "error", "detail": str(error)})
            finally:
                finished.set()

        worker = Thread(target=work, name="chat-generation", daemon=True)
        worker.start()
        try:
            yield json.dumps({"type": "status", "text": "Загрузка модели и генерация"}) + "\n"
            while not finished.is_set() or not queue.empty():
                try:
                    payload = await asyncio.to_thread(queue.get, True, 1.0)
                except Empty:
                    yield '{"type":"heartbeat"}\n'
                    continue
                yield json.dumps(payload, ensure_ascii=False) + "\n"
        finally:
            closed.set()

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
