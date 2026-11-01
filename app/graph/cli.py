"""`uv run ask`: answer one question through the graph and show how it got there."""

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from app.config import get_settings
from app.db import make_engine
from app.graph.build import build_question_graph
from app.graph.state import Citation, TraceStep
from app.models import ModelConfigError
from app.retrieval.bm25 import MissingSnapshotError
from app.retrieval.search import Search


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ask", description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--version", default="latest", help="a dated version, latest or all")
    return parser.parse_args(argv)


def format_sources(citations: list[Citation]) -> str:
    lines = ["Sources"]
    for c in citations:
        lines.append(f"  [{c['number']}] {c['spec_version']}  {c['heading_path']}")
        lines.append(f"      {c['source_url']}")
    return "\n".join(lines)


def format_trace(trace: list[TraceStep]) -> str:
    width = max(len(step["step"]) for step in trace)
    lines = ["How I got this answer"]
    for step in trace:
        lines.append(
            f"  {step['step']:<{width}}  {step['ms'] / 1000:5.1f}s  {step['tokens']:>5} tok"
            f"  {step['summary']}"
        )
    total_ms = sum(step["ms"] for step in trace)
    total_tokens = sum(step["tokens"] for step in trace)
    lines.append(f"  total {total_ms / 1000:.1f}s, {total_tokens} tokens")
    return "\n".join(lines)


def format_result(result: dict[str, Any]) -> str:
    parts = [result["answer"].strip()]
    if result.get("citations"):
        parts.append(format_sources(result["citations"]))
    parts.append(format_trace(result["trace"]))
    return "\n\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    engine = make_engine(settings.database_url)
    try:
        search = Search.from_settings(settings, engine)
        search.scope(args.version)  # reject unknown versions before any model call
        graph = build_question_graph(settings, search)
        result = graph.invoke({"question": args.question, "spec_version": args.version})
    except (MissingSnapshotError, ModelConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    print(format_result(result))
    return 0
