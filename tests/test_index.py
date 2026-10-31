import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.indexing import InMemoryRecordManager
from langchain_core.vectorstores import InMemoryVectorStore
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.db import make_engine
from ingest.index import build_record_manager, build_vector_store, index_chunks


def chunk(content: str, source: str = "docs/a.mdx", version: str = "2026-07-28") -> Document:
    return Document(page_content=content, metadata={"source_path": source, "spec_version": version})


@pytest.fixture
def stores() -> tuple[InMemoryRecordManager, InMemoryVectorStore]:
    manager = InMemoryRecordManager("test")
    manager.create_schema()
    return manager, InMemoryVectorStore(DeterministicFakeEmbedding(size=8))


def test_reindexing_skips_unchanged_and_removes_stale(
    stores: tuple[InMemoryRecordManager, InMemoryVectorStore],
) -> None:
    manager, store = stores
    corpus = [chunk("one"), chunk("two"), chunk("three", source="docs/b.mdx")]

    first = index_chunks(corpus, manager, store)
    second = index_chunks(corpus, manager, store)
    changed = index_chunks([chunk("one"), chunk("two edited")], manager, store)

    assert (first["num_added"], first["num_skipped"]) == (3, 0)
    assert (second["num_added"], second["num_skipped"], second["num_deleted"]) == (0, 3, 0)
    # "two" was edited and docs/b.mdx disappeared upstream: both old chunks are deleted.
    assert (changed["num_added"], changed["num_skipped"], changed["num_deleted"]) == (1, 1, 2)
    assert sorted(record["text"] for record in store.store.values()) == ["one", "two edited"]


def test_same_text_in_two_versions_is_two_records(
    stores: tuple[InMemoryRecordManager, InMemoryVectorStore],
) -> None:
    manager, store = stores

    result = index_chunks(
        [chunk("same", version="2025-11-25"), chunk("same", version="2026-07-28")], manager, store
    )

    assert result["num_added"] == 2


def test_progress_reported_every_batch(
    stores: tuple[InMemoryRecordManager, InMemoryVectorStore],
) -> None:
    manager, store = stores
    seen: list[int] = []

    index_chunks([chunk(f"text {i}") for i in range(250)], manager, store, seen.append)

    assert seen == [100, 200]


@pytest.mark.db
def test_pgvector_round_trip_with_version_filter() -> None:
    engine = make_engine()
    collection = f"test_{uuid.uuid4().hex[:8]}"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        engine.dispose()
        pytest.skip("database not reachable")

    try:
        store = build_vector_store(engine, DeterministicFakeEmbedding(size=8), 8, collection)
        manager = build_record_manager(engine, collection)
        docs = [
            chunk("sampling requests", version="2025-11-25"),
            chunk("sampling requests", version="2026-07-28", source="docs/b.mdx"),
            chunk("guides", version="unversioned", source="docs/c.mdx"),
        ]

        first = index_chunks(docs, manager, store)
        second = index_chunks(docs, manager, store)
        version_filter = {"spec_version": {"$in": ["2026-07-28", "unversioned"]}}
        hits = store.similarity_search("sampling requests", k=5, filter=version_filter)

        assert first["num_added"] == 3
        assert second["num_skipped"] == 3
        assert {h.metadata["spec_version"] for h in hits} == {"2026-07-28", "unversioned"}
    finally:
        store.delete_collection()
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM upsertion_record WHERE namespace = :ns"),
                {"ns": f"pgvector/{collection}"},
            )
        engine.dispose()
