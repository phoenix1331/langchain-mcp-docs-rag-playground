"""`uv run evals`: run the question set through the graph, score it and save the results.

Results go to `evals/results/<UTC timestamp>-<label>.json`, rewritten after every
question so an interrupted run can continue with `--resume`. Free-tier model quotas
need a pause between questions (`EVAL_PAUSE_SECONDS` or `--pause`); per-minute rate
limits are retried, a daily quota stops the run cleanly.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from langchain_core.runnables import Runnable

from app.config import Settings, get_settings
from app.db import make_engine
from app.graph.build import build_question_graph
from app.models import chat_model
from app.retrieval.search import Search
from evals.langsmith_sync import publish
from evals.questions import Question, load_questions
from evals.scoring import JUDGE_PROMPT, Judge, Judgement, make_judge, score, summarise

RESULTS_DIR = Path(__file__).parent / "results"
RETRY_SECONDS = 60.0
MAX_RETRIES = 3


class QuestionGraph(Protocol):
    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]: ...


class DailyQuotaExceeded(RuntimeError):
    """A provider's daily quota is used up: waiting a minute will not help."""


def is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return any(marker in text for marker in ("RateLimit", "429", "RESOURCE_EXHAUSTED"))


def with_retries[T](
    call: Callable[[], T], sleep: Callable[[float], None], log: Callable[[str], None]
) -> T:
    attempt = 0
    while True:
        try:
            return call()
        except Exception as exc:
            if not is_rate_limit(exc):
                raise
            if "PerDay" in str(exc):
                raise DailyQuotaExceeded(str(exc)[:300]) from exc
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            log(f"  rate limited; waiting {RETRY_SECONDS:.0f}s (retry {attempt})")
            sleep(RETRY_SECONDS)


def build_judge(settings: Settings) -> Runnable[dict[str, str], dict[str, Any]]:
    """`JUDGE_MODEL`, falling back to `ANSWER_MODEL`, with structured output."""
    llm = chat_model(settings.judge_model or settings.answer_model)
    return JUDGE_PROMPT | llm.with_structured_output(Judgement, include_raw=True)


def config_summary(settings: Settings) -> dict[str, Any]:
    return {
        "router_model": settings.router_model,
        "answer_model": settings.answer_model,
        "judge_model": settings.judge_model or settings.answer_model,
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "mode": "hybrid",
        "rerank": True,
        "grading": True,
        "top_k": settings.top_k,
        "rerank_top_n": settings.rerank_top_n,
        "relevance_threshold": settings.relevance_threshold,
        "max_rewrites": settings.max_rewrites,
    }


def run_question(
    question: Question,
    graph: QuestionGraph,
    judge: Judge,
    sleep: Callable[[float], None],
    log: Callable[[str], None],
) -> dict[str, Any]:
    base = {
        "id": question.id,
        "category": question.category,
        "expected": question.expected,
        "spec_version": question.spec_version,
        "question": question.question,
    }
    inputs = {"question": question.question, "spec_version": question.spec_version}
    try:
        final = with_retries(lambda: graph.invoke(inputs), sleep, log)
        scored = with_retries(lambda: score(question, final, judge), sleep, log)
    except DailyQuotaExceeded:
        raise
    except Exception as exc:
        return {**base, "error": f"{type(exc).__name__}: {exc}"[:500], "passed": False}

    trace = final.get("trace", [])
    return {
        **base,
        "error": None,
        "answer": final.get("answer", ""),
        "steps": [step["step"] for step in trace],
        "model_calls": sum(1 for step in trace if step["tokens"]),
        "seconds": round(sum(step["ms"] for step in trace) / 1000, 1),
        "tokens": sum(step["tokens"] for step in trace),
        **scored,
    }


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def status(result: dict[str, Any]) -> str:
    if result["error"]:
        return f"  error: {result['error'][:150]}"
    verdict = "pass" if result["passed"] else "FAIL"
    parts = [verdict, result["outcome"]]
    if result["facts_found"]:
        parts.append(f"facts {sum(result['facts_found'])}/{len(result['facts_found'])}")
    if result["retrieval_hit"] is not None:
        parts.append("retrieved" if result["retrieval_hit"] else "missed sources")
    parts.append(f"{result['seconds']}s")
    return "  " + ", ".join(parts)


def run(
    questions: list[Question],
    graph: QuestionGraph,
    judge: Judge,
    payload: dict[str, Any],
    path: Path,
    pause: float,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every question not already in `payload` without error; save after each one."""
    results = [r for r in payload.get("results", []) if r.get("error") is None]
    done = {r["id"] for r in results}
    payload.update(results=results, complete=False)
    payload.pop("stopped", None)
    todo = [q for q in questions if q.id not in done]

    for index, question in enumerate(todo):
        if index and pause:
            sleep(pause)
        log(f"[{len(results) + 1}/{len(questions)}] {question.id}: {question.question}")
        try:
            result = run_question(question, graph, judge, sleep, log)
        except DailyQuotaExceeded as exc:
            payload["stopped"] = f"daily quota reached: {exc}"
            save(path, payload)
            log(f'daily quota reached; continue later with: make evals ARGS="--resume {path}"')
            return payload
        results.append(result)
        payload["summary"] = summarise(results)
        save(path, payload)
        log(status(result))

    order = {q.id: n for n, q in enumerate(questions)}
    results.sort(key=lambda r: order.get(r["id"], len(order)))
    payload.update(complete=True, summary=summarise(results))
    save(path, payload)
    return payload


def format_summary(summary: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value:.0%}"

    return "\n".join(
        [
            f"questions          {summary['questions']} ({summary['errors']} errors)",
            f"pass rate          {pct(summary['pass_rate'])}",
            f"retrieval hit rate {pct(summary['retrieval_hit_rate'])}",
            f"correctness        {pct(summary['correctness'])}",
            f"citation accuracy  {pct(summary['citation_accuracy'])}",
            f"correct declines   {pct(summary['correct_declines'])}",
            f"false declines     {summary['false_declines']}",
            f"direct replies     {pct(summary['direct_correct'])}",
            f"mean time          {summary['mean_seconds']}s per question",
            f"tokens             {summary['total_tokens']}",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="evals", description=__doc__)
    parser.add_argument("--label", default="run", help="name for this run, e.g. baseline")
    parser.add_argument("--ids", help="comma-separated question ids to run")
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    parser.add_argument("--resume", type=Path, help="a results file to continue")
    parser.add_argument("--pause", type=float, help="seconds between questions")
    parser.add_argument("--no-langsmith", dest="langsmith", action="store_false")
    return parser.parse_args(argv)


def select(questions: list[Question], ids: str | None, limit: int | None) -> list[Question]:
    if ids:
        wanted = {i.strip() for i in ids.split(",")}
        unknown = wanted - {q.id for q in questions}
        if unknown:
            raise ValueError(f"unknown question ids: {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q.id in wanted]
    return questions[:limit] if limit else questions


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    try:
        questions = select(load_questions(), args.ids, args.limit)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.resume or RESULTS_DIR / f"{stamp}-{args.label}.json"
    payload: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8"))
        if args.resume
        else {"label": args.label, "started_at": stamp, "config": config_summary(settings)}
    )

    engine = make_engine(settings.database_url)
    try:
        search = Search.from_settings(settings, engine)
        graph = build_question_graph(settings, search)
        judge = make_judge(build_judge(settings))
        pause = settings.eval_pause_seconds if args.pause is None else args.pause
        payload = run(questions, graph, judge, payload, path, pause)
    finally:
        engine.dispose()

    print(f"\nresults: {path}")
    print(format_summary(payload["summary"]))
    if not payload["complete"]:
        return 3
    experiment = publish(payload, questions) if args.langsmith else None
    if experiment:
        print(f"LangSmith experiment: {experiment}")
    return 0
