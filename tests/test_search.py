from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_classic.retrievers import ContextualCompressionRetriever

from app.config import Settings
from app.retrieval import search as search_module
from app.retrieval.bm25 import KeywordIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.search import Mode, Search
from tests.retrieval_fakes import ReverseReranker, StaticRetriever, doc

CHUNKS = [
    doc("old", "2025-11-25", "Lifecycle › Initialization"),
    doc("new", "2026-07-28", "Lifecycle › Initialization"),
    doc("guide", "unversioned", "Guides › Initialization"),
]


@pytest.fixture
def search() -> Search:
    store = MagicMock()
    store.as_retriever.return_value = StaticRetriever(docs=[CHUNKS[1]])
    return Search(keyword=KeywordIndex(CHUNKS), store=store, reranker=ReverseReranker(), top_k=5)


def test_hybrid_rerank_is_the_default(search: Search) -> None:
    retriever = search.retriever("latest")

    assert isinstance(retriever, ContextualCompressionRetriever)
    assert isinstance(retriever.base_retriever, HybridRetriever)
    hits = retriever.invoke("initialization")
    assert {d.metadata["spec_version"] for d in hits} <= {"2026-07-28", "unversioned"}
    assert all("relevance_score" in d.metadata for d in hits)


def test_vector_scope_reaches_the_store(search: Search) -> None:
    search.retriever("2025-11-25", Mode.VECTOR, rerank=False).invoke("q")

    search.store.as_retriever.assert_called_with(  # type: ignore[attr-defined]
        search_kwargs={"k": 5, "filter": {"spec_version": {"$in": ["2025-11-25", "unversioned"]}}}
    )


def test_single_modes_without_rerank(search: Search) -> None:
    bm25 = search.retriever("all", Mode.BM25, rerank=False)

    assert {d.id for d in bm25.invoke("initialization")} == {"old", "new", "guide"}
    assert search.retriever("all", Mode.VECTOR, rerank=False).invoke("q")[0].id == "new"


def test_scope_and_unknown_versions(search: Search) -> None:
    assert search.scope("all") is None
    with pytest.raises(ValueError, match="unknown spec version"):
        search.retriever("1999-01-01")


def test_from_settings_wires_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://u:p@localhost/db",
        router_model="fake:r",
        answer_model="fake:a",
        embed_model="fake:e",
        rerank_model="fake:rr",
        top_k=9,
        cache_dir=tmp_path,
    )
    monkeypatch.setattr(search_module.KeywordIndex, "load", lambda cache_dir: "keyword")
    monkeypatch.setattr(search_module, "query_vector_store", lambda s, e: "store")
    monkeypatch.setattr(search_module, "build_reranker", lambda spec, n: f"{spec}/{n}")

    built = Search.from_settings(settings, MagicMock())

    assert (built.keyword, built.store, built.reranker, built.top_k) == (
        "keyword",
        "store",
        "fake:rr/6",
        9,
    )
