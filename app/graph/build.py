"""Graph wiring. Nodes are passed in, so tests can swap in fakes."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings
from app.graph.nodes import (
    Node,
    build_grader,
    build_router,
    make_answer_node,
    make_decline_node,
    make_grade_node,
    make_retrieve_node,
    make_rewrite_node,
    make_route_node,
)
from app.graph.state import GraphInput, GraphState, Route
from app.models import chat_model
from app.retrieval.search import Search

ROUTE_TARGETS: dict[Route, str] = {
    "retrieve": "retrieve",
    "direct": "answer",
    "out_of_scope": "decline",
}


@dataclass(frozen=True)
class Nodes:
    route: Node
    retrieve: Node
    grade: Node
    rewrite: Node
    answer: Node
    decline: Node


def after_route(state: GraphState) -> Route:
    return state["route"]


GradeOutcome = Literal["answer", "rewrite", "decline"]


def make_after_grade(
    relevance_threshold: int, max_rewrites: int
) -> Callable[[GraphState], GradeOutcome]:
    """Enough relevant chunks: answer. Too few: rewrite, up to the cap; then decline."""

    def after_grade(state: GraphState) -> GradeOutcome:
        if sum(state.get("grades", [])) >= relevance_threshold:
            return "answer"
        if state.get("rewrites", 0) < max_rewrites:
            return "rewrite"
        return "decline"

    return after_grade


def build_graph(
    nodes: Nodes, relevance_threshold: int = 2, max_rewrites: int = 2
) -> CompiledStateGraph:
    graph = StateGraph(GraphState, input_schema=GraphInput)
    graph.add_node("route", nodes.route)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("rewrite", nodes.rewrite)
    graph.add_node("answer", nodes.answer)
    graph.add_node("decline", nodes.decline)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route, ROUTE_TARGETS)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        make_after_grade(relevance_threshold, max_rewrites),
        ["answer", "rewrite", "decline"],
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", END)
    graph.add_edge("decline", END)
    return graph.compile()


def build_question_graph(settings: Settings, search: Search) -> CompiledStateGraph:
    """The real graph: `ROUTER_MODEL` routes, grades and rewrites; hybrid + rerank search;
    `ANSWER_MODEL` answers. Thresholds come from settings."""
    versions = search.keyword.versions
    return build_graph(
        Nodes(
            route=make_route_node(build_router(settings)),
            retrieve=make_retrieve_node(search),
            grade=make_grade_node(build_grader(settings)),
            rewrite=make_rewrite_node(chat_model(settings.router_model)),
            answer=make_answer_node(chat_model(settings.answer_model), versions),
            decline=make_decline_node(versions),
        ),
        relevance_threshold=settings.relevance_threshold,
        max_rewrites=settings.max_rewrites,
    )
