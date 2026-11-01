"""Graph nodes. Each takes the state and returns a partial update."""

import re
import time
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from app.config import Settings
from app.graph.state import Citation, GraphState, Route, trace_step
from app.models import chat_model
from app.retrieval.scope import ALL
from app.retrieval.search import Mode, Search
from ingest.fetch import LATEST, SpecVersions

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


def make_retrieve_node(search: Search, mode: Mode = Mode.HYBRID, rerank: bool = True) -> Node:
    """Search the selected version (plus unversioned pages) for the current query."""

    def retrieve(state: GraphState) -> dict[str, Any]:
        started = time.monotonic()
        docs = search.retriever(state["spec_version"], mode, rerank).invoke(state["query"])
        how = f"{mode} + rerank" if rerank else str(mode)
        summary = f"{len(docs)} chunks for {state['query']!r} ({how})"
        return {"documents": docs, "trace": [trace_step("retrieve", summary, started)]}

    return retrieve


ANSWER_SYSTEM = """\
You answer questions about the Model Context Protocol (MCP) using only the numbered \
sources below, taken from the official documentation.

Rules:
- Use only facts stated in the sources. Do not add outside knowledge.
- Cite every factual sentence with its source number in square brackets, e.g. [2] or \
[1][3]. Only cite numbers listed below.
- The user asked about spec version {version}. Sources are labelled with their version; \
`unversioned` pages apply to every version. If versions disagree, say which says what.
- Sources are also labelled Specification, Guide or Extension. Only Specification \
sources define what a version supports. Guides are copied across versions and may \
describe features added later: if a feature appears only in Guide sources, say the \
specification sources for this version do not mention it. Extensions are optional \
additions to the protocol.
- If the sources do not answer the question, say so plainly and mention what they do \
cover. Do not guess.
- Lead with the direct answer, then supporting detail. Use Markdown; put message, field \
and method names in backticks.

Sources:

{sources}"""

DIRECT_SYSTEM = """\
You are the assistant of a question-answering tool for the Model Context Protocol (MCP) \
documentation at modelcontextprotocol.io. This message needs no documentation lookup: \
it is a greeting, thanks, or a question about you. Reply in one or two friendly \
sentences. If useful, explain that you answer questions about the MCP specification, \
guides, SDKs, Inspector and extensions for a chosen spec version, citing the source \
sections. Do not state facts about MCP itself."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ANSWER_SYSTEM), ("human", "{question}")]
)
DIRECT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", DIRECT_SYSTEM), ("human", "{question}")]
)
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def relevant_documents(state: GraphState) -> list[Document]:
    """Graded-relevant chunks once grading has run on these documents; otherwise all."""
    docs = state.get("documents", [])
    grades = state.get("grades")
    if grades is None or len(grades) != len(docs):
        return docs
    return [doc for doc, relevant in zip(docs, grades, strict=True) if relevant]


def describe_version(version: str, versions: SpecVersions) -> str:
    if version == ALL:
        return "all (any version)"
    if version == LATEST:
        return f"latest ({versions.latest})"
    return version


SOURCE_KINDS = {"Specification": "Specification", "Extensions": "Extension"}


def source_kind(doc: Document) -> str:
    """Label from the docs site tab; everything outside spec and extensions is a guide."""
    return SOURCE_KINDS.get(doc.metadata.get("nav_tab", ""), "Guide")


def format_sources(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{n}] {d.metadata['heading_path']} ({d.metadata['spec_version']}, {source_kind(d)})"
        f"\n{d.page_content}"
        for n, d in enumerate(docs, start=1)
    )


def cited(answer: str, sources: list[Document]) -> list[Citation]:
    """Sources referenced as `[n]`, `[n][m]` or `[n, m]`, in number order."""
    numbers = {int(n) for group in CITATION_PATTERN.findall(answer) for n in group.split(",")}
    return [
        Citation(
            number=n,
            chunk_id=sources[n - 1].id,
            heading_path=sources[n - 1].metadata["heading_path"],
            spec_version=sources[n - 1].metadata["spec_version"],
            source_url=sources[n - 1].metadata["source_url"],
        )
        for n in sorted(numbers)
        if 1 <= n <= len(sources)
    ]


def make_answer_node(llm: Runnable[Any, BaseMessage], versions: SpecVersions) -> Node:
    """Direct route: a short reply with no sources. Otherwise answer from the sources."""
    answer_chain = ANSWER_PROMPT | llm
    direct_chain = DIRECT_PROMPT | llm

    def answer(state: GraphState) -> dict[str, Any]:
        started = time.monotonic()
        if state.get("route") == "direct":
            reply = direct_chain.invoke({"question": state["question"]})
            return {
                "answer": reply.text,
                "citations": [],
                "trace": [trace_step("answer", "direct reply, no retrieval", started, reply)],
            }

        sources = relevant_documents(state)
        reply = answer_chain.invoke(
            {
                "question": state["question"],
                "version": describe_version(state["spec_version"], versions),
                "sources": format_sources(sources),
            }
        )
        citations = cited(reply.text, sources)
        summary = f"cited {len(citations)} of {len(sources)} sources"
        return {
            "answer": reply.text,
            "citations": citations,
            "trace": [trace_step("answer", summary, started, reply)],
        }

    return answer


OUT_OF_SCOPE_ANSWER = (
    "That is outside what I can help with: I only answer questions about the Model "
    "Context Protocol, from its documentation at modelcontextprotocol.io."
)


def make_decline_node() -> Node:
    def decline(state: GraphState) -> dict[str, Any]:
        started = time.monotonic()
        return {
            "answer": OUT_OF_SCOPE_ANSWER,
            "citations": [],
            "trace": [trace_step("decline", "out of scope", started)],
        }

    return decline
