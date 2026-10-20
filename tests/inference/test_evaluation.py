"""Очередь checkpoints и отчёт проверяются без моделей, GPU и HTTP-запросов."""

from pathlib import Path
from typing import cast

import pytest
import torch

from mini_llm.inference.config import GenerationConfig
from mini_llm.inference.evaluation import pipeline
from mini_llm.inference.evaluation.report import literal_block
from mini_llm.inference.evaluation.schemas import QUESTIONS, EvaluationProgress, EvaluationRequest
from mini_llm.inference.evaluation.session import CheckpointSession
from mini_llm.inference.schemas import GenerationResult
from mini_llm.pretrained import PreparedPretrained, PretrainedConfig
from mini_llm.pretrained.model import HuggingFaceCausalLMBackend
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer
from mini_llm.workspace.catalog import ModelRun


@pytest.fixture
def evaluation_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelRun:
    monkeypatch.chdir(tmp_path)
    directory = Path("checkpoints/runs/custom/2ch/v5/run-001")
    directory.mkdir(parents=True)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    run = ModelRun(
        run_id="custom--2ch-v5--run-001",
        model_id="custom",
        dataset_id="2ch",
        dataset_version="v5",
        directory=directory,
        title="Тестовый запуск",
        tokenizer_file=tokenizer,
    )
    run.save()
    for name in ("step_00000050.pt", "step_00000100.pt"):
        (directory / name).write_bytes(b"fake weights")
    return run


class FakeSession:
    """Одна сессия; вместо inference вернуть значение из изолированного RNG."""

    def __init__(self, run: ModelRun, config: GenerationConfig) -> None:
        self.loaded: list[str] = []
        self.calls: list[tuple[str, str]] = []
        self.config = config
        self.closed = False

    def load(self, checkpoint: Path) -> None:
        self.loaded.append(checkpoint.name)

    def generate(self, prompt: str) -> GenerationResult:
        self.calls.append((self.loaded[-1], prompt))
        return GenerationResult(f"answer-{torch.randint(10000, (1,)).item()}", [1, 2])

    def close(self) -> None:
        self.closed = True


def test_all_checkpoints_get_all_questions_in_order_with_one_session(
    evaluation_run: ModelRun,
) -> None:
    snapshots: list[EvaluationProgress] = []
    sessions: list[FakeSession] = []
    for step in range(150, 651, 50):
        (evaluation_run.directory / f"step_{step:08d}.pt").write_bytes(b"fake weights")

    def factory(run: ModelRun, config: GenerationConfig) -> FakeSession:
        session = FakeSession(run, config)
        sessions.append(session)
        return session

    original_files = {p: p.read_bytes() for p in evaluation_run.directory.glob("*.pt")}
    rng = torch.get_rng_state().clone()
    result = pipeline.run_evaluation(
        EvaluationRequest(run_id=evaluation_run.run_id, device="cpu"),
        report_progress=snapshots.append,
        session_factory=factory,
    )
    assert len(sessions) == 1 and sessions[0].closed
    assert sessions[0].config.max_new_tokens == 1024
    assert sessions[0].calls == [(name, q) for name in sessions[0].loaded for q in QUESTIONS]
    assert result["completed"] == result["total"] == 65
    assert snapshots[-1].checkpoints_completed == 13
    assert snapshots[-1].eta_seconds == 0
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    answers = [
        s.latest_result.answer for s in snapshots if s.stage == "Ответ сохранён" and s.latest_result
    ]
    assert all(answers[:5] == answers[start : start + 5] for start in range(0, 65, 5))
    report = Path(str(result["report_file"])).read_text(encoding="utf-8")
    assert report.count("\n### Ответ\n") == 65
    assert all(question in report for question in QUESTIONS)
    assert snapshots[-1].latest_result is not None
    assert snapshots[-1].latest_result.question == QUESTIONS[-1]
    assert all(p.read_bytes() == value for p, value in original_files.items())


def test_stop_preserves_partial_report_and_next_run_never_overwrites_it(
    evaluation_run: ModelRun,
) -> None:
    snapshots: list[EvaluationProgress] = []
    request = EvaluationRequest(run_id=evaluation_run.run_id, device="cpu")
    result = pipeline.run_evaluation(
        request,
        report_progress=snapshots.append,
        session_factory=FakeSession,
        should_stop=lambda: bool(snapshots and snapshots[-1].completed >= 2),
    )
    path = Path(str(result["report_file"]))
    previous = path.read_bytes()
    assert result["interrupted"] and result["completed"] == 2
    assert path.read_text(encoding="utf-8").count("\n### Ответ\n") == 2
    second = pipeline.run_evaluation(request, session_factory=FakeSession)
    assert second["report_file"] != result["report_file"] and path.read_bytes() == previous


def test_failure_keeps_answers_closes_session_and_records_error(evaluation_run: ModelRun) -> None:
    request = EvaluationRequest(run_id=evaluation_run.run_id, device="cpu")
    config = GenerationConfig(device="cpu")

    class Failing(FakeSession):
        def generate(self, prompt: str) -> GenerationResult:
            if len(self.calls) == 2:
                raise RuntimeError("synthetic OOM; no retry")
            return super().generate(prompt)

    session = Failing(evaluation_run, config)
    snapshots: list[EvaluationProgress] = []
    with pytest.raises(RuntimeError, match="no retry"):
        pipeline.run_evaluation(
            request, session_factory=lambda *_: session, report_progress=snapshots.append
        )
    assert session.closed and len(session.calls) == 2
    report = Path(snapshots[-1].report_file).read_text(encoding="utf-8")
    assert report.count("\n### Ответ\n") == 2
    assert "synthetic OOM" in report and "Ошибка" in snapshots[-1].stage


def test_stop_before_first_checkpoint_never_initializes_model(evaluation_run: ModelRun) -> None:
    def forbidden(run: ModelRun, config: GenerationConfig) -> FakeSession:
        pytest.fail("Остановленная queued-задача не должна создавать модель")

    result = pipeline.run_evaluation(
        EvaluationRequest(run_id=evaluation_run.run_id),
        should_stop=lambda: True,
        session_factory=forbidden,
    )
    assert result["completed"] == 0 and result["interrupted"]


def test_plan_counts_aliases_and_rejects_missing_or_unsafe_runs(evaluation_run: ModelRun) -> None:
    (evaluation_run.directory / "last.pt").write_bytes(b"alias")
    (evaluation_run.directory / "best.pt").write_bytes(b"alias")
    _, checkpoints = pipeline.evaluation_plan(EvaluationRequest(run_id=evaluation_run.run_id))
    assert [p.name for p in checkpoints] == [
        "step_00000050.pt",
        "step_00000100.pt",
        "best.pt",
        "last.pt",
    ]
    with pytest.raises(ValueError, match="не найден"):
        pipeline.evaluation_plan(EvaluationRequest(run_id="unknown"))
    with pytest.raises(ValueError):
        EvaluationRequest(run_id="../escape")
    text = "before\n```\n<script>test</script>\n````\nafter"
    block = literal_block(text)
    assert block.startswith("`````text\n") and text in block and block.endswith("`````\n")


def test_pretrained_session_loads_base_once_and_switches_complete_states(
    evaluation_run: ModelRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mini_llm.inference.evaluation import session as module

    base = PretrainedConfig(model_id="local/tiny", torch_dtype="float32")
    config_file = evaluation_run.directory / "pretrained.json"
    base.save(config_file)
    run = evaluation_run.model_copy(
        update={"model_id": base.model_id, "pretrained_config_file": config_file}
    )
    backend = HuggingFaceCausalLMBackend(torch.nn.Linear(2, 2), base)
    prepared = PreparedPretrained(backend, cast(HuggingFaceDialogueTokenizer, object()))
    calls: list[PretrainedConfig] = []

    def prepare(config: PretrainedConfig, *, for_inference: bool) -> PreparedPretrained:
        assert for_inference
        calls.append(config)
        return prepared

    monkeypatch.setattr(module, "prepare_pretrained_model", prepare)
    generator = CheckpointSession(run, GenerationConfig(device="cpu"))
    for value in (1.0, 2.0):
        checkpoint = evaluation_run.directory / f"step-{value}.pt"
        torch.save(
            {
                "model_metadata": backend.checkpoint_metadata,
                "model_state_dict": {
                    k: torch.full_like(v, value) for k, v in backend.module.state_dict().items()
                },
            },
            checkpoint,
        )
        generator.load(checkpoint)
        assert all(torch.all(t == value) for t in backend.module.state_dict().values())
        assert not backend.module.training
    assert len(calls) == 1
    bad = evaluation_run.directory / "bad.pt"
    torch.save({"model_metadata": backend.checkpoint_metadata, "model_state_dict": {}}, bad)
    with pytest.raises(ValueError, match="неполный"):
        generator.load(bad)
    generator.close()
    assert generator.prepared is None
