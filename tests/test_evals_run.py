import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from evals import run as run_module
from evals.questions import Question, load_questions
from evals.run import (
    DailyQuotaExceeded,
    config_summary,
    format_summary,
    is_transient,
    main,
    run,
    run_question,
    select,
    status,
    with_retries,
)

QUESTIONS = load_questions()
SETTINGS = SimpleNamespace(
    router_model="fake:r",
    answer_model="fake:a",
    judge_model=None,
    embed_model="fake:e",
    rerank_model="fake:rr",
    top_k=20,
    rerank_top_n=6,
    relevance_threshold=2,
    max_rewrites=2,
)
BY_ID = {q.id: q for q in QUESTIONS}


class FakeGraph:
    """Answers retrieve questions with the expected source; declines out-of-scope ones."""

    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.failures = failures or []
        self.calls: list[dict[str, str]] = []

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        self.calls.append(inputs)
        if self.failures:
            raise self.failures.pop(0)
        q = next(q for q in QUESTIONS if q.question == inputs["question"])
        step = {"summary": "", "ms": 1000, "tokens": 100, "details": []}
        if q.expected == "decline":
            trace = [{"step": "route", **step}, {"step": "decline", **step, "tokens": 0}]
            return {"route": "out_of_scope", "trace": trace, "answer": "no"}
        docs = [Document(page_content="x", metadata={"source_path": s}) for s in q.sources]
        trace = [{"step": s, **step} for s in ("route", "retrieve", "grade", "answer")]
        citations = [{"source_path": s} for s in q.sources[:1]]
        return {
            "route": "retrieve",
            "trace": trace,
            "documents": docs,
            "citations": citations,
            "answer": "yes",
        }


def judge_all(q: Question, answer: str) -> tuple[list[bool], str, int]:
    return [True] * len(q.key_facts), "ok", 10


def no_sleep(seconds: float) -> None:
    pass


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RuntimeError("429 RESOURCE_EXHAUSTED"), True),
        (type("GoogleRateLimitError", (Exception,), {})("slow down"), True),
        (RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand"), True),
        (ValueError("bad input"), False),
    ],
)
def test_is_transient(exc: Exception, expected: bool) -> None:
    assert is_transient(exc) is expected


def test_with_retries_waits_then_succeeds() -> None:
    attempts = iter([RuntimeError("429"), RuntimeError("429"), "ok"])
    sleeps: list[float] = []
    logs: list[str] = []

    def call() -> str:
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    assert with_retries(call, sleeps.append, logs.append) == "ok"
    assert sleeps == [60.0, 60.0]
    assert logs[0] == "  rate limited or overloaded; waiting 60s (retry 1)"


def test_with_retries_gives_up_and_passes_other_errors() -> None:
    def always_limited() -> None:
        raise RuntimeError("429")

    def broken() -> None:
        raise ValueError("nope")

    with pytest.raises(RuntimeError):
        with_retries(always_limited, no_sleep, lambda m: None)
    with pytest.raises(ValueError):
        with_retries(broken, no_sleep, lambda m: None)


def test_daily_quota_is_not_retried() -> None:
    def exhausted() -> None:
        raise RuntimeError("429 GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(DailyQuotaExceeded):
        with_retries(exhausted, no_sleep, lambda m: None)


def test_run_question_records_cost_and_scores() -> None:
    result = run_question(BY_ID["discover-purpose"], FakeGraph(), judge_all, no_sleep, print)

    assert result["error"] is None
    assert result["passed"] is True
    assert (result["seconds"], result["tokens"], result["model_calls"]) == (4.0, 400, 4)
    assert result["steps"] == ["route", "retrieve", "grade", "answer"]


def test_run_question_records_errors() -> None:
    graph = FakeGraph(failures=[ValueError("graph exploded")])

    result = run_question(BY_ID["discover-purpose"], graph, judge_all, no_sleep, print)

    assert result["error"] == "ValueError: graph exploded"
    assert result["passed"] is False


def test_run_saves_after_each_question_and_completes(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    questions = [BY_ID["discover-purpose"], BY_ID["banana-bread"]]
    sleeps: list[float] = []
    logs: list[str] = []

    graph = FakeGraph()

    payload = run(questions, graph, judge_all, {"label": "t"}, path, 5, sleeps.append, logs.append)

    saved = json.loads(path.read_text())
    assert saved["complete"] is True
    assert [r["id"] for r in saved["results"]] == ["discover-purpose", "banana-bread"]
    assert saved["summary"]["pass_rate"] == 1.0
    assert payload == saved
    assert sleeps == [5]  # paused between questions, not before the first
    assert logs[0].startswith("[1/2] discover-purpose:")
    assert logs[1] == "  pass, answered, facts 3/3, retrieved, 4.0s"


def test_resume_skips_done_questions_and_retries_errors(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    done = run_question(BY_ID["discover-purpose"], FakeGraph(), judge_all, no_sleep, print)
    failed = {"id": "banana-bread", "error": "boom", "passed": False}
    graph = FakeGraph()
    questions = [BY_ID["discover-purpose"], BY_ID["banana-bread"]]
    previous = {"results": [done, failed]}

    payload = run(questions, graph, judge_all, previous, path, 0, no_sleep, print)

    assert [c["question"] for c in graph.calls] == [BY_ID["banana-bread"].question]
    assert [r["id"] for r in payload["results"]] == ["discover-purpose", "banana-bread"]
    assert payload["results"][1]["error"] is None


def test_daily_quota_stops_the_run_and_keeps_progress(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    quota = RuntimeError("429 GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    graph = FakeGraph()
    questions = [BY_ID["discover-purpose"], BY_ID["banana-bread"]]
    logs: list[str] = []
    original = graph.invoke

    def invoke(inputs: dict[str, str]) -> dict[str, Any]:
        if len(graph.calls) == 1:
            raise quota
        return original(inputs)

    graph.invoke = invoke  # type: ignore[method-assign]

    payload = run(questions, graph, judge_all, {}, path, 0, no_sleep, logs.append)

    assert payload["complete"] is False
    assert payload["stopped"].startswith("daily quota reached")
    assert [r["id"] for r in json.loads(path.read_text())["results"]] == ["discover-purpose"]
    assert f"--resume {path}" in logs[-1]


def test_status_lines() -> None:
    assert status({"error": "boom"}) == "  error: boom"
    decline = {
        "error": None,
        "passed": True,
        "outcome": "declined_out_of_scope",
        "facts_found": [],
        "retrieval_hit": None,
        "seconds": 1.0,
    }
    assert status(decline) == "  pass, declined_out_of_scope, 1.0s"
    missed = {**decline, "passed": False, "outcome": "answered", "retrieval_hit": False}
    assert status(missed) == "  FAIL, answered, missed sources, 1.0s"


def test_select() -> None:
    assert [q.id for q in select(QUESTIONS, "banana-bread, greeting", None)] == [
        "banana-bread",
        "greeting",
    ]
    assert len(select(QUESTIONS, None, 3)) == 3
    with pytest.raises(ValueError, match="unknown question ids: nope"):
        select(QUESTIONS, "nope", None)


def test_format_summary() -> None:
    summary = {
        "questions": 46,
        "errors": 1,
        "pass_rate": 0.8,
        "retrieval_hit_rate": 0.9,
        "correctness": None,
        "citation_accuracy": 0.75,
        "correct_declines": 1.0,
        "false_declines": 2,
        "direct_correct": 1.0,
        "mean_seconds": 5.2,
        "total_tokens": 150000,
    }

    text = format_summary(summary)

    assert "questions          46 (1 errors)" in text
    assert "pass rate          80%" in text
    assert "correctness        -" in text
    assert "false declines     2" in text


def test_config_summary_falls_back_to_the_answer_model() -> None:
    settings = MagicMock(judge_model=None, answer_model="google_genai:a")

    assert config_summary(settings)["judge_model"] == "google_genai:a"


@pytest.fixture
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    settings = MagicMock(database_url="db", eval_pause_seconds=0, judge_model=None)
    engine = MagicMock()
    published: list[str] = []
    monkeypatch.setattr(run_module, "get_settings", lambda: settings)
    monkeypatch.setattr(run_module, "make_engine", lambda url: engine)
    monkeypatch.setattr(run_module.Search, "from_settings", lambda s, e: MagicMock())
    graphs: list[tuple[Any, ...]] = []

    def build_graph(settings: Any, search: Any, *switches: Any) -> FakeGraph:
        graphs.append(switches)
        return FakeGraph()

    monkeypatch.setattr(run_module, "build_question_graph", build_graph)
    monkeypatch.setattr(run_module, "build_judge", lambda s: MagicMock())
    monkeypatch.setattr(run_module, "make_judge", lambda chain: judge_all)
    monkeypatch.setattr(run_module, "RESULTS_DIR", tmp_path)
    real_config = run_module.config_summary
    monkeypatch.setattr(
        run_module, "config_summary", lambda s, *switches: real_config(SETTINGS, *switches)
    )
    monkeypatch.setattr(
        run_module, "publish", lambda payload, qs: published.append(payload["label"]) or "exp-1"
    )
    return {"engine": engine, "published": published, "dir": tmp_path, "graphs": graphs}


def test_main_runs_saves_and_publishes(
    wired: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--label", "baseline", "--ids", "banana-bread"]) == 0

    [path] = list(wired["dir"].glob("*-baseline.json"))
    assert json.loads(path.read_text())["config"]
    out = capsys.readouterr().out
    assert "pass rate          100%" in out
    assert "LangSmith experiment: exp-1" in out
    assert wired["published"] == ["baseline"]
    wired["engine"].dispose.assert_called_once()


def test_main_skips_langsmith_when_asked(wired: dict[str, Any]) -> None:
    assert main(["--ids", "banana-bread", "--no-langsmith"]) == 0
    assert wired["published"] == []


def test_main_resumes_an_existing_file(wired: dict[str, Any]) -> None:
    path = wired["dir"] / "old.json"
    config = {"mode": "vector", "rerank": False, "grading": False}
    path.write_text(json.dumps({"label": "old", "config": config, "results": []}))

    # The resumed run keeps its stored switches, whatever the command line says.
    assert main(["--resume", str(path), "--ids", "banana-bread", "--no-langsmith"]) == 0
    assert json.loads(path.read_text())["config"] == config
    assert wired["graphs"] == [("vector", False, False)]


def test_main_returns_3_when_stopped(
    wired: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_module, "run", lambda *a, **k: {"complete": False, "summary": {
        "questions": 0, "errors": 0, "pass_rate": None, "retrieval_hit_rate": None,
        "correctness": None, "citation_accuracy": None, "correct_declines": None,
        "false_declines": 0, "direct_correct": None, "mean_seconds": None, "total_tokens": 0,
    }})  # fmt: skip

    assert main(["--ids", "banana-bread"]) == 3
    assert wired["published"] == []


def test_main_rejects_unknown_ids(
    wired: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--ids", "nope"]) == 1
    assert "unknown question ids: nope" in capsys.readouterr().err


def test_build_judge_uses_the_judge_or_answer_model(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[str] = []

    def fake_chat_model(spec: str) -> MagicMock:
        built.append(spec)
        return MagicMock()

    monkeypatch.setattr(run_module, "chat_model", fake_chat_model)

    run_module.build_judge(MagicMock(judge_model=None, answer_model="google_genai:a"))
    run_module.build_judge(MagicMock(judge_model="anthropic:j", answer_model="google_genai:a"))

    assert built == ["google_genai:a", "anthropic:j"]


def test_main_passes_config_switches_to_the_graph(wired: dict[str, Any]) -> None:
    args = ["--ids", "banana-bread", "--mode", "vector", "--no-rerank", "--no-grading"]

    assert main([*args, "--no-langsmith"]) == 0

    assert wired["graphs"] == [("vector", False, False)]
    [path] = list(wired["dir"].glob("*-run.json"))
    config = json.loads(path.read_text())["config"]
    assert (config["mode"], config["rerank"], config["grading"]) == ("vector", False, False)
