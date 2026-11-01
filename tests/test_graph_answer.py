from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.graph import build as build_module
from app.graph.build import build_question_graph
from app.graph.nodes import (
    Grades,
    RouteDecision,
    cited,
    describe_chunk,
    describe_version,
    format_sources,
    make_answer_node,
    make_retrieve_node,
    relevant_documents,
    source_kind,
)
from app.graph.state import GraphState
from app.retrieval.bm25 import KeywordIndex
from app.retrieval.search import Mode, Search
from ingest.fetch import SpecVersions
from tests.retrieval_fakes import ReverseReranker, StaticRetriever, doc

VERSIONS = SpecVersions(("2025-11-25", "2026-07-28"))
SOURCES = [doc("a", heading="Auth › Refresh"), doc("b", "unversioned", "Ext › Creds"), doc("c")]


class FakeLLM:
    """Records the prompt it was given and replies with fixed text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[ChatPromptValue] = []

    def runnable(self) -> RunnableLambda:
        def reply(prompt: ChatPromptValue) -> AIMessage:
            self.prompts.append(prompt)
            usage = {"input_tokens": 90, "output_tokens": 10, "total_tokens": 100}
            return AIMessage(content=self.text, usage_metadata=usage)

        return RunnableLambda(reply)

    def system(self) -> str:
        return str(self.prompts[-1].messages[0].content)


def state(**extra: Any) -> GraphState:
    return {"question": "How are tokens refreshed?", "spec_version": "latest", **extra}  # type: ignore[typeddict-item]


@pytest.fixture
def search() -> Search:
    store = MagicMock()
    store.as_retriever.return_value = StaticRetriever(docs=[SOURCES[0]])
    return Search(keyword=KeywordIndex(SOURCES), store=store, reranker=ReverseReranker(), top_k=5)


def test_retrieve_searches_the_current_query(search: Search) -> None:
    update = make_retrieve_node(search)(state(query="refresh token"))

    assert update["documents"]
    assert all("relevance_score" in d.metadata for d in update["documents"])
    [step] = update["trace"]
    assert step["step"] == "retrieve"
    assert "for 'refresh token' (hybrid + rerank)" in step["summary"]
    assert len(step["details"]) == len(update["documents"])
    assert all(" · rerank " in detail for detail in step["details"])


def test_describe_chunk_includes_scores_and_ranks_when_present() -> None:
    plain = doc("a", heading="Auth › Refresh")
    extra = {"relevance_score": 0.714, "ranks": {"bm25": 4, "vector": 1}}
    scored = plain.model_copy(update={"metadata": {**plain.metadata, **extra}})

    assert describe_chunk(plain) == "Auth › Refresh (2026-07-28)"
    expected = "Auth › Refresh (2026-07-28) · rerank 0.71 · bm25 #4, vector #1"
    assert describe_chunk(scored) == expected


def test_retrieve_mode_and_rerank_are_configurable(search: Search) -> None:
    update = make_retrieve_node(search, Mode.VECTOR, rerank=False)(state(query="q"))

    assert [d.id for d in update["documents"]] == ["a"]
    assert update["trace"][0]["summary"].endswith("(vector)")


@pytest.mark.parametrize(
    ("answer", "numbers"),
    [
        ("Refresh it [1]. Credentials differ [2][3].", [1, 2, 3]),
        ("Both say so [3, 1].", [1, 3]),
        ("Twice [2] and again [2].", [2]),
        ("Out of range [0][4] and none.", []),
    ],
)
def test_cited_parses_citation_forms(answer: str, numbers: list[int]) -> None:
    assert [c["number"] for c in cited(answer, SOURCES)] == numbers


def test_citation_carries_source_details() -> None:
    [citation] = cited("Yes [2].", SOURCES)

    assert citation == {
        "number": 2,
        "chunk_id": "b",
        "heading_path": "Ext › Creds",
        "spec_version": "unversioned",
        "source_url": "https://example.test/b",
        "source_path": "docs/b.mdx",
    }


def test_relevant_documents_uses_grades_only_when_they_match() -> None:
    assert relevant_documents(state(documents=SOURCES)) == SOURCES
    assert relevant_documents(state(documents=SOURCES, grades=[True, False, True])) == [
        SOURCES[0],
        SOURCES[2],
    ]
    # Grades left over from an earlier retrieval are ignored.
    assert relevant_documents(state(documents=SOURCES, grades=[False])) == SOURCES
    assert relevant_documents(state()) == []


@pytest.mark.parametrize(
    ("version", "label"),
    [("latest", "latest (2026-07-28)"), ("all", "all (any version)"), ("2025-11-25", "2025-11-25")],
)
def test_describe_version(version: str, label: str) -> None:
    assert describe_version(version, VERSIONS) == label


def test_format_sources_numbers_and_labels_each_chunk() -> None:
    text = format_sources(SOURCES[:2])

    assert text.startswith("[1] Auth › Refresh (2026-07-28, Guide)\nAuth › Refresh\n\nbody of a")
    assert "\n\n[2] Ext › Creds (unversioned, Guide)\n" in text


@pytest.mark.parametrize(
    ("tab", "kind"),
    [("Specification", "Specification"), ("Extensions", "Extension"), ("Documentation", "Guide")],
)
def test_source_kind_comes_from_the_docs_tab(tab: str, kind: str) -> None:
    chunk = doc("x")
    chunk.metadata["nav_tab"] = tab

    assert source_kind(chunk) == kind


def test_answer_uses_only_the_relevant_sources_and_records_citations() -> None:
    llm = FakeLLM("Clients refresh with a grant [1]. Not this [2].")
    node = make_answer_node(llm.runnable(), VERSIONS)

    update = node(state(route="retrieve", documents=SOURCES, grades=[False, True, True]))

    system = llm.system()
    assert "spec version latest (2026-07-28)" in system
    assert "body of a" not in system  # graded irrelevant
    assert "[1] Ext › Creds (unversioned, Guide)" in system
    assert update["answer"] == "Clients refresh with a grant [1]. Not this [2]."
    assert [(c["number"], c["chunk_id"]) for c in update["citations"]] == [(1, "b"), (2, "c")]
    assert update["trace"][0]["tokens"] == 100
    assert update["trace"][0]["summary"] == "cited 2 of 2 sources"


def test_direct_route_answers_without_sources() -> None:
    llm = FakeLLM("Hello! Ask me about MCP.")

    update = make_answer_node(llm.runnable(), VERSIONS)(state(route="direct", question="hi"))

    assert "needs no documentation lookup" in llm.system()
    assert update["answer"] == "Hello! Ask me about MCP."
    assert update["citations"] == []
    assert update["trace"][0]["summary"] == "direct reply, no retrieval"


def test_build_question_graph_wires_real_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, search: Search
) -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://u:p@localhost/db",
        router_model="fake:router",
        answer_model="fake:answer",
        embed_model="fake:e",
        rerank_model="fake:rr",
        relevance_threshold=1,  # the fake search finds one chunk for this question
        cache_dir=tmp_path,
    )
    route_reply = {
        "raw": AIMessage(content=""),
        "parsed": RouteDecision(route="retrieve", reason="about MCP"),
    }
    built_models: list[str] = []
    grade_reply = {"raw": AIMessage(content=""), "parsed": Grades(relevant=[1, 2], missing="")}
    router = RunnableLambda(lambda inputs: route_reply)
    grader = RunnableLambda(lambda inputs: grade_reply)
    monkeypatch.setattr(build_module, "build_router", lambda s: router)
    monkeypatch.setattr(build_module, "build_grader", lambda s: grader)

    def fake_chat_model(spec: str) -> RunnableLambda:
        built_models.append(spec)
        return FakeLLM("Use the refresh grant [1].").runnable()

    monkeypatch.setattr(build_module, "chat_model", fake_chat_model)

    result = build_question_graph(settings, search).invoke(state())

    assert built_models == ["fake:router", "fake:answer"]  # rewrite uses the fast model
    assert [s["step"] for s in result["trace"]] == ["route", "retrieve", "grade", "answer"]
    assert result["answer"] == "Use the refresh grant [1]."
    assert len(result["citations"]) == 1
