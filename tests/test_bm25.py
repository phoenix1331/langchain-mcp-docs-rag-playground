import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.retrieval.bm25 import (
    SNAPSHOT_NAME,
    BM25Retriever,
    KeywordIndex,
    MissingSnapshotError,
    read_snapshot,
    stored_chunks,
    tokenize,
    write_snapshot,
)


def doc(content: str, version: str = "2026-07-28", key: str | None = None) -> Document:
    return Document(id=key or content, page_content=content, metadata={"spec_version": version})


CORPUS = [
    doc("The client sends initialize to start the lifecycle", "2025-11-25", "init-old"),
    doc("The client sends initialize to start the lifecycle", "2026-07-28", "init-new"),
    doc("Servers expose tools via tools/list", "2026-07-28", "tools"),
    doc("Use the MCP Inspector to debug a server", "unversioned", "inspector"),
    doc("Sampling lets servers request completions", "2026-07-28", "sampling"),
]


def test_tokenize_lowercases_and_splits_on_punctuation() -> None:
    tokens = tokenize("Call `tools/list`, then tools_call!")

    assert tokens == ["call", "tools", "list", "then", "tools_call"]


def test_retriever_ranks_by_bm25_and_drops_non_matches() -> None:
    retriever = BM25Retriever(docs=CORPUS, k=3)

    hits = retriever.invoke("tools list")

    assert [h.id for h in hits] == ["tools"]


def test_retriever_respects_k() -> None:
    hits = BM25Retriever(docs=CORPUS, k=1).invoke("client initialize servers")

    assert len(hits) == 1


def test_terms_in_most_documents_still_score() -> None:
    docs = [doc("server", key="1"), doc("server", key="2"), doc("client", key="3")]

    assert [h.id for h in BM25Retriever(docs=docs, k=5).invoke("server")] == ["1", "2"]


def test_empty_retriever_returns_nothing() -> None:
    assert BM25Retriever(docs=[]).invoke("anything") == []


def test_keyword_index_scopes_and_caches() -> None:
    index = KeywordIndex(CORPUS)

    scoped = index.retriever(("2025-11-25", "unversioned"), k=5)

    assert {h.id for h in scoped.invoke("initialize inspector")} == {"init-old", "inspector"}
    assert {h.id for h in index.retriever(None, k=5).invoke("initialize")} == {
        "init-old",
        "init-new",
    }
    assert index.retriever(("2025-11-25", "unversioned"), k=5) is scoped
    assert index.versions.versions == ("2025-11-25", "2026-07-28")


def test_snapshot_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / SNAPSHOT_NAME

    write_snapshot(CORPUS, path)
    loaded = KeywordIndex.load(path.parent)

    assert [(d.id, d.page_content, d.metadata) for d in loaded.chunks] == [
        (d.id, d.page_content, d.metadata) for d in CORPUS
    ]
    assert not path.with_suffix(".tmp").exists()
    assert json.loads(path.read_text())[0]["id"] == "init-old"


def test_missing_snapshot_explains_the_fix(tmp_path: Path) -> None:
    with pytest.raises(MissingSnapshotError, match="run `make ingest` first"):
        read_snapshot(tmp_path / SNAPSHOT_NAME)


def test_stored_chunks_reads_the_collection() -> None:
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.all.return_value = [("id-1", "text", {"spec_version": "unversioned"})]

    chunks = stored_chunks(engine, "mcp_docs")

    expected = Document(id="id-1", page_content="text", metadata={"spec_version": "unversioned"})
    assert chunks == [expected]
    assert conn.execute.call_args.args[1] == {"name": "mcp_docs"}
