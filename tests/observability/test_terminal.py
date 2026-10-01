"""Быстрые тесты формата terminal logging и throttling."""

import pytest

from mini_llm.observability import ProgressThrottle, format_elapsed, terminal_log, terminal_stage


def test_terminal_stage_reports_start_and_finish(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with terminal_stage("TEST", "короткий этап"):
        terminal_log("TEST", "работа выполняется", elapsed=1.2)

    output = capsys.readouterr().out
    assert "[TEST] Начало: короткий этап" in output
    assert "[TEST] работа выполняется | elapsed=00:01" in output
    assert "[TEST] Завершено: короткий этап" in output


def test_progress_throttle_does_not_log_every_item() -> None:
    throttle = ProgressThrottle(every_items=100, every_seconds=60.0)

    assert throttle.should_report(1)
    assert not throttle.should_report(2)
    assert throttle.should_report(101)
    assert format_elapsed(3_661) == "1:01:01"
