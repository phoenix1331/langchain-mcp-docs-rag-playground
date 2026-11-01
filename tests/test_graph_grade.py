from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.graph.build import Nodes, build_graph, make_after_grade
from app.graph.nodes import (
    CLOSEST_SECTIONS,
    OUT_OF_SCOPE_ANSWER,
    Grades,
    make_decline_node,
    make_grade_node,
    make_rewrite_node,
)
from app.graph.progress import describe_result, describe_start
from app.graph.state import GraphState
from ingest.fetch import SpecVersions
from tests.retrieval_fakes import doc

VERSIONS = SpecVersions(("2025-11-25", "2026-07-28"))
DOCS = [doc("a", heading="Auth › Refresh"), doc("b"), doc("c"), doc("d")]
USAGE = {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55}


def state(**extra: Any) -> GraphState:
    base = {"question": "How are tokens refreshed?", "spec_version": "latest", "query": "q"}
    return {**base, **extra}  # type: ignore[typeddict-item]


def grader(parsed: Grades | None, seen: list[dict[str, str]] | None = None) -> RunnableLambda:
    def reply(inputs: dict[str, str]) -> dict[str, Any]:
        if seen is not None:
            seen.append(inputs)
        return {"raw": AIMessage(content="", usage_metadata=USAGE), "parsed": parsed}

    return RunnableLambda(reply)


def test_grade_marks_each_chunk_in_one_call() -> None:
    seen: list[dict[str, str]] = []
    node = make_grade_node(grader(Grades(relevant=[1, 3, 9], missing="the HTTP request"), seen))

    update = node(state(documents=DOCS))

    assert len(seen) == 1
    assert "[4] Base › Page (2026-07-28, Guide)" in seen[0]["sources"]
    assert update["grades"] == [True, False, True, False]
    assert update["missing"] == "the HTTP request"
    [step] = update["trace"]
    assert step["summary"] == "2 of 4 chunks relevant; missing: the HTTP request"
    assert step["tokens"] == 55


def test_grade_keeps_everything_when_the_reply_is_unparseable() -> None:
    update = make_grade_node(grader(None))(state(documents=DOCS))

    assert update["grades"] == [True] * 4
    assert update["missing"] == ""
    assert "unparseable, so all kept" in update["trace"][0]["summary"]


def test_grade_skips_the_model_without_documents() -> None:
    seen: list[dict[str, str]] = []

    update = make_grade_node(grader(Grades(relevant=[], missing=""), seen))(state(documents=[]))

    assert seen == []
    assert (update["grades"], update["missing"]) == ([], "no results")


def rewriter(text: str, seen: list[str]) -> RunnableLambda:
    def reply(prompt: Any) -> AIMessage:
        seen.append(prompt.messages[1].content)
        return AIMessage(content=text, usage_metadata=USAGE)

    return RunnableLambda(reply)


def test_rewrite_uses_what_was_missing_and_counts() -> None:
    seen: list[str] = []
    node = make_rewrite_node(rewriter('"refresh_token grant_types client metadata"\n', seen))

    update = node(state(query="refresh tokens", missing="client metadata", rewrites=1))

    assert "Previous query: refresh tokens\nWhat was missing: client metadata" in seen[0]
    assert update["query"] == "refresh_token grant_types client metadata"
    assert update["rewrites"] == 2
    assert update["trace"][0]["summary"] == "rewrite 2: 'refresh_token grant_types client metadata'"


def test_empty_rewrite_falls_back_to_the_question() -> None:
    update = make_rewrite_node(rewriter("  ", []))(state())

    assert update["query"] == "How are tokens refreshed?"
    assert update["rewrites"] == 1


def test_decline_out_of_scope() -> None:
    update = make_decline_node(VERSIONS)(state(route="out_of_scope"))

    assert update["answer"] == OUT_OF_SCOPE_ANSWER
    assert update["citations"] == []


def test_decline_not_covered_lists_the_closest_sections() -> None:
    update = make_decline_node(VERSIONS)(state(route="retrieve", documents=DOCS, rewrites=2))

    assert update["answer"].startswith(
        "The MCP docs for latest (2026-07-28) don't cover this: 3 searches found"
    )
    assert "- [1] Auth › Refresh (2026-07-28)" in update["answer"]
    assert [c["chunk_id"] for c in update["citations"]] == ["a", "b", "c"]
    assert len(update["citations"]) == CLOSEST_SECTIONS
    assert update["trace"][0]["summary"] == "not covered after 3 searches; 3 closest sections"


def test_decline_without_any_results() -> None:
    update = make_decline_node(VERSIONS)(state(route="retrieve"))

    assert "1 search found too few" in update["answer"]
    assert "closest" not in update["answer"]
    assert update["citations"] == []


@pytest.mark.parametrize(
    ("grades", "rewrites", "outcome"),
    [
        ([True, True, False], 0, "answer"),
        ([True, False, False], 0, "rewrite"),
        ([True, False, False], 1, "rewrite"),
        ([True, False, False], 2, "decline"),
        ([], 2, "decline"),
    ],
)
def test_after_grade(grades: list[bool], rewrites: int, outcome: str) -> None:
    after_grade = make_after_grade(relevance_threshold=2, max_rewrites=2)

    assert after_grade(state(grades=grades, rewrites=rewrites)) == outcome


def scripted_graph(grade_rounds: list[list[bool]], max_rewrites: int = 2) -> Any:
    """Graph whose grade node returns the next scripted round; records the visit order."""
    visited: list[str] = []
    rounds = iter(grade_rounds)

    def step(name: str, update: dict[str, Any] | None = None) -> Any:
        def node(graph_state: GraphState) -> dict[str, Any]:
            visited.append(name)
            extra = update(graph_state) if callable(update) else (update or {})
            return {**extra, "trace": [{"step": name, "summary": "", "ms": 0, "tokens": 0}]}

        return node

    graph = build_graph(
        Nodes(
            route=step("route", {"route": "retrieve", "query": "q", "rewrites": 0}),
            retrieve=step("retrieve", {"documents": DOCS[:3]}),
            grade=step("grade", lambda s: {"grades": next(rounds)}),
            rewrite=step("rewrite", lambda s: {"rewrites": s["rewrites"] + 1}),
            answer=step("answer", {"answer": "ok"}),
            decline=step("decline", {"answer": "no"}),
        ),
        relevance_threshold=2,
        max_rewrites=max_rewrites,
    )
    return graph, visited


def test_graph_rewrites_until_enough_chunks_are_relevant() -> None:
    graph, visited = scripted_graph([[False] * 3, [True, True, False]])

    final = graph.invoke({"question": "q", "spec_version": "latest"})

    assert visited == ["route", "retrieve", "grade", "rewrite", "retrieve", "grade", "answer"]
    assert final["rewrites"] == 1


def test_graph_declines_after_the_rewrite_cap() -> None:
    graph, visited = scripted_graph([[False] * 3] * 3)

    final = graph.invoke({"question": "q", "spec_version": "latest"})

    assert visited.count("rewrite") == 2
    assert visited[-1] == "decline"
    assert final["answer"] == "no"


def test_zero_rewrites_declines_straight_after_grading() -> None:
    graph, visited = scripted_graph([[False] * 3], max_rewrites=0)

    graph.invoke({"question": "q", "spec_version": "latest"})

    assert visited == ["route", "retrieve", "grade", "decline"]


@pytest.mark.parametrize(
    ("step", "extra", "message"),
    [
        ("route", {}, "Deciding whether this needs the docs..."),
        ("retrieve", {"query": "refresh"}, "Searching the latest docs for 'refresh'..."),
        ("grade", {"documents": DOCS}, "Checking which of the 4 chunks are relevant..."),
        ("rewrite", {}, "Too few relevant chunks, so rephrasing the search..."),
        ("answer", {"route": "direct"}, "Replying..."),
        (
            "answer",
            {"route": "retrieve", "documents": DOCS, "grades": [True, False, True, False]},
            "Writing the answer from 2 relevant chunks...",
        ),
        ("decline", {}, "Explaining why this can't be answered..."),
        ("other", {}, "Running other..."),
    ],
)
def test_describe_start(step: str, extra: dict[str, Any], message: str) -> None:
    assert describe_start(step, {**state(), **extra}) == message


def test_describe_result() -> None:
    trace = [{"step": "grade", "summary": "2 of 6 chunks relevant", "ms": 1040, "tokens": 1780}]

    expected = "  done in 1.0s, 1780 tokens: 2 of 6 chunks relevant"

    assert describe_result({"trace": trace}) == expected
    no_tokens = [{**trace[0], "tokens": 0}]
    assert describe_result({"trace": no_tokens}) == "  done in 1.0s: 2 of 6 chunks relevant"
    assert describe_result({}) == "  done"
