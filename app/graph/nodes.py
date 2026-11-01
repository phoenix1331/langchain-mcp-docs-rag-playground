"""Graph nodes. Each takes the state and returns a partial update."""

import time
from collections.abc import Callable
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from app.config import Settings
from app.graph.state import GraphState, Route, trace_step
from app.models import chat_model

Node = Callable[[GraphState], dict[str, Any]]


class RouteDecision(BaseModel):
    route: Route = Field(
        description=(
            "retrieve: needs the MCP docs. direct: greeting, thanks or a question about "
            "this assistant. out_of_scope: not about the Model Context Protocol."
        )
    )
    reason: str = Field(description="One short sentence explaining the choice.")


ROUTE_SYSTEM = """\
You route questions for an assistant that answers from the Model Context Protocol (MCP) \
documentation at modelcontextprotocol.io: the specification (lifecycle, transports, \
authorization, tools, resources, prompts, sampling, roots, elicitation, utilities, \
schema), the guides and tutorials, SDKs, the Inspector, and MCP extensions.

Choose one route:
- retrieve: anything about MCP, its concepts, messages, versions or ecosystem, including \
comparisons with other technologies. When unsure, choose retrieve.
- direct: greetings, thanks, small talk, or questions about what this assistant can do.
- out_of_scope: questions with no connection to MCP (general coding help, other \
products, world knowledge)."""

ROUTE_PROMPT = ChatPromptTemplate.from_messages([("system", ROUTE_SYSTEM), ("human", "{question}")])

FALLBACK_DECISION = RouteDecision(
    route="retrieve", reason="the router reply could not be parsed, so search to be safe"
)


def build_router(settings: Settings) -> Runnable[dict[str, str], dict[str, Any]]:
    """Prompt | `ROUTER_MODEL` with structured output; `include_raw` keeps token usage."""
    llm = chat_model(settings.router_model)
    return ROUTE_PROMPT | llm.with_structured_output(RouteDecision, include_raw=True)


def make_route_node(router: Runnable[dict[str, str], dict[str, Any]]) -> Node:
    def route(state: GraphState) -> dict[str, Any]:
        started = time.monotonic()
        reply = router.invoke({"question": state["question"]})
        decision: RouteDecision = reply["parsed"] or FALLBACK_DECISION
        return {
            "route": decision.route,
            "query": state["question"],
            "rewrites": 0,
            "trace": [
                trace_step("route", f"{decision.route}: {decision.reason}", started, reply["raw"])
            ],
        }

    return route
