from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.graph.build import ROUTE_TARGETS, Nodes, build_graph
from app.graph.nodes import (
    GRADE_PROMPT,
    ROUTE_PROMPT,
    RouteDecision,
    build_grader,
    build_router,
    make_route_node,
)
from app.graph.state import GraphState, trace_step


def router_reply(route: str | None, tokens: int = 42) -> RunnableLambda:
    raw = AIMessage(
        content="",
        usage_metadata={"input_tokens": tokens - 2, "output_tokens": 2, "total_tokens": tokens},
    )
    parsed = RouteDecision(route=route, reason="because") if route else None  # type: ignore[arg-type]
    return RunnableLambda(lambda inputs: {"raw": raw, "parsed": parsed, "parsing_error": None})


def state(question: str = "What is sampling?") -> GraphState:
    return {"question": question, "spec_version": "latest"}


@pytest.mark.parametrize("route", ["retrieve", "direct", "out_of_scope"])
def test_route_node_records_decision_query_and_trace(route: str) -> None:
    update = make_route_node(router_reply(route))(state())

    assert update["route"] == route
    assert update["query"] == "What is sampling?"
    assert update["rewrites"] == 0
    [step] = update["trace"]
    assert (step["step"], step["summary"], step["tokens"]) == ("route", f"{route}: because", 42)
    assert step["ms"] >= 0


def test_unparseable_router_reply_falls_back_to_retrieve() -> None:
    update = make_route_node(router_reply(None))(state())

    assert update["route"] == "retrieve"
    assert "could not be parsed" in update["trace"][0]["summary"]


def test_router_receives_the_question() -> None:
    seen: list[dict[str, str]] = []

    def record(inputs: dict[str, str]) -> dict[str, Any]:
        seen.append(inputs)
        return router_reply("direct").invoke(inputs)

    make_route_node(RunnableLambda(record))(state("hello"))

    assert seen == [{"question": "hello"}]


def test_trace_step_without_usage_has_zero_tokens_and_no_details() -> None:
    assert trace_step("x", "y", 0.0)["tokens"] == 0
    assert trace_step("x", "y", 0.0)["details"] == []
    assert trace_step("x", "y", 0.0, details=["a"])["details"] == ["a"]
    assert trace_step("x", "y", 0.0, AIMessage(content="no usage"))["tokens"] == 0


def visit(name: str, visited: list[str], update: dict[str, Any] | None = None) -> Any:
    def node(graph_state: GraphState) -> dict[str, Any]:
        visited.append(name)
        return {**(update or {}), "trace": [{"step": name, "summary": "", "ms": 0, "tokens": 0}]}

    return node


@pytest.mark.parametrize(
    ("route", "path"),
    [
        ("retrieve", ["retrieve", "grade", "answer"]),
        ("direct", ["answer"]),
        ("out_of_scope", ["decline"]),
    ],
)
def test_graph_follows_the_route(route: str, path: list[str]) -> None:
    visited: list[str] = []
    graph = build_graph(
        Nodes(
            route=make_route_node(router_reply(route)),
            retrieve=visit("retrieve", visited),
            grade=visit("grade", visited, {"grades": [True, True]}),
            rewrite=visit("rewrite", visited),
            answer=visit("answer", visited),
            decline=visit("decline", visited),
        )
    )

    final = graph.invoke(state())

    assert visited == path
    assert final["route"] == route
    assert [s["step"] for s in final["trace"]] == ["route", *path]


def test_every_route_has_a_target() -> None:
    assert set(ROUTE_TARGETS) == set(RouteDecision.model_fields["route"].annotation.__args__)  # type: ignore[union-attr]


def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql+psycopg://u:p@localhost/db",
        router_model="google_genai:flash-test",
        answer_model="fake:a",
        embed_model="fake:e",
        rerank_model="fake:rr",
        cache_dir=tmp_path,
    )


def test_build_router_chains_prompt_and_configured_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")

    router = build_router(settings(tmp_path))

    assert router.first is ROUTE_PROMPT  # type: ignore[attr-defined]
    assert "flash-test" in repr(router)


def test_build_grader_uses_the_fast_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")

    grader = build_grader(settings(tmp_path))

    assert grader.first is GRADE_PROMPT  # type: ignore[attr-defined]
    assert "flash-test" in repr(grader)
