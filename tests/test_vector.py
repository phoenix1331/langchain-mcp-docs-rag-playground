import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.config import Settings, get_settings
from app.db import make_engine
from app.retrieval import vector
from app.retrieval.bm25 import stored_chunks
from app.retrieval.vector import query_vector_store, scope_filter, vector_retriever
from ingest.index import build_vector_store


def test_scope_filter() -> None:
    assert scope_filter(None) is None
    assert scope_filter(("2026-07-28", "unversioned")) == {
        "spec_version": {"$in": ["2026-07-28", "unversioned"]}
    }


def test_vector_retriever_passes_k_and_filter() -> None:
    store = MagicMock()

    vector_retriever(store, ("2026-07-28", "unversioned"), k=7)

    store.as_retriever.assert_called_once_with(
        search_kwargs={"k": 7, "filter": {"spec_version": {"$in": ["2026-07-28", "unversioned"]}}}
    )


def test_query_vector_store_uses_the_configured_model_and_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://u:p@localhost/db",
        router_model="fake:r",
        answer_model="fake:a",
        embed_model="voyageai:voyage-test",
        rerank_model="fake:rr",
        embed_dimensions=16,
        cache_dir=tmp_path,
    )
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(vector, "embedding_model", lambda spec, dims: f"{spec}/{dims}")
    monkeypatch.setattr(vector, "build_vector_store", lambda *args: calls.append(args))

    query_vector_store(settings, "engine")  # type: ignore[arg-type]
    query_vector_store(settings, "engine", "fake")  # type: ignore[arg-type]

    collection = "mcp_docs-voyageai-voyage-test-16"
    assert calls == [
        ("engine", "voyageai:voyage-test/16", 16, collection),
        ("engine", "fake", 16, collection),
    ]


@pytest.mark.db
def test_scoped_vector_search_and_stored_chunks_against_pgvector() -> None:
    engine = make_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        engine.dispose()
        pytest.skip("database not reachable")

    dims = get_settings().embed_dimensions
    collection = f"test_{uuid.uuid4().hex[:8]}"
    store = build_vector_store(engine, DeterministicFakeEmbedding(size=dims), dims, collection)
    try:
        store.add_documents(
            [
                Document(page_content="sampling", metadata={"spec_version": "2025-11-25"}),
                Document(page_content="sampling", metadata={"spec_version": "2026-07-28"}),
                Document(page_content="sampling", metadata={"spec_version": "unversioned"}),
            ],
            ids=["a", "b", "c"],
        )

        scoped = vector_retriever(store, ("2026-07-28", "unversioned"), k=5).invoke("sampling")
        everything = vector_retriever(store, None, k=5).invoke("sampling")
        stored = stored_chunks(engine, collection)

        assert {d.metadata["spec_version"] for d in scoped} == {"2026-07-28", "unversioned"}
        assert len(everything) == 3
        assert [d.id for d in stored] == ["a", "b", "c"]
    finally:
        store.delete_collection()
        engine.dispose()
