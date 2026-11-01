"""Graph wiring. Nodes are passed in, so tests can swap in fakes."""

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings
from app.graph.nodes import (
    Node,
    build_router,
    make_answer_node,
    make_decline_node,
    make_retrieve_node,
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
    answer: Node
    decline: Node


def after_route(state: GraphState) -> Route:
    return state["route"]


def build_graph(nodes: Nodes) -> CompiledStateGraph:
    graph = StateGraph(GraphState, input_schema=GraphInput)
    graph.add_node("route", nodes.route)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("answer", nodes.answer)
    graph.add_node("decline", nodes.decline)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route, ROUTE_TARGETS)
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("decline", END)
    return graph.compile()


def build_question_graph(settings: Settings, search: Search) -> CompiledStateGraph:
    """The graph with real nodes: `ROUTER_MODEL`, hybrid + rerank search, `ANSWER_MODEL`."""
    return build_graph(
        Nodes(
            route=make_route_node(build_router(settings)),
            retrieve=make_retrieve_node(search),
            answer=make_answer_node(chat_model(settings.answer_model), search.keyword.versions),
            decline=make_decline_node(),
        )
    )
