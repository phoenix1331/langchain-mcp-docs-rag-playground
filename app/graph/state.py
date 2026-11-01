"""Question graph state, and the per-step trace shown in the "How I got this answer" panel."""

import operator
import time
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

Route = Literal["retrieve", "direct", "out_of_scope"]


class TraceStep(TypedDict):
    step: str
    summary: str
    ms: int
    tokens: int


class Citation(TypedDict):
    number: int  # the [n] used in the answer text
    chunk_id: str | None
    heading_path: str
    spec_version: str
    source_url: str


class GraphInput(TypedDict):
    question: str
    spec_version: str  # a dated version, `latest` or `all`


class GraphState(GraphInput):
    route: NotRequired[Route]
    query: NotRequired[str]  # the search query; starts as the question, may be rewritten
    rewrites: NotRequired[int]
    documents: NotRequired[list[Document]]
    grades: NotRequired[list[bool]]  # relevance per entry in `documents`
    answer: NotRequired[str]
    citations: NotRequired[list[Citation]]  # sources the answer actually cites
    trace: NotRequired[Annotated[list[TraceStep], operator.add]]  # nodes append, never replace


def trace_step(step: str, summary: str, started: float, message: object = None) -> TraceStep:
    """A trace entry timed from `started` (`time.monotonic()`), with tokens from an LLM reply."""
    tokens = 0
    if isinstance(message, AIMessage) and message.usage_metadata:
        tokens = message.usage_metadata["total_tokens"]
    return TraceStep(
        step=step,
        summary=summary,
        ms=round((time.monotonic() - started) * 1000),
        tokens=tokens,
    )
