"""Graph wiring. Nodes are passed in, so tests can swap in fakes."""

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import Node
from app.graph.state import GraphInput, GraphState, Route

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
