"""Keyword search: a BM25 retriever over a JSON snapshot of the stored chunks.

The ingest writes the snapshot from pgvector after indexing, so BM25 sees exactly the
chunks vector search sees. `rank_bm25` does the scoring; LangChain's own `BM25Retriever`
lives in the sunset `langchain-community` package and is a thin wrapper, so this module
provides the retriever directly.
"""

import json
import math
import os
import re
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
from rank_bm25 import BM25Okapi
from sqlalchemy import Engine, text

from app.retrieval.scope import Scope, in_scope, stored_versions
from ingest.fetch import SpecVersions

SNAPSHOT_NAME = "bm25-chunks.json"
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


class MissingSnapshotError(RuntimeError):
    """The BM25 snapshot has not been written yet."""


def tokenize(content: str) -> list[str]:
    """Lowercase word tokens; `tools/list` and `notifications/initialized` split on `/`."""
    return TOKEN_PATTERN.findall(content.lower())


def stored_chunks(engine: Engine, collection: str) -> list[Document]:
    """Every chunk PGVector holds for `collection`, in a stable order."""
    query = text(
        "SELECT e.id, e.document, e.cmetadata FROM langchain_pg_embedding e "
        "JOIN langchain_pg_collection c ON c.uuid = e.collection_id "
        "WHERE c.name = :name ORDER BY e.id"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"name": collection}).all()
    return [Document(id=row[0], page_content=row[1], metadata=row[2]) for row in rows]


def write_snapshot(chunks: list[Document], path: Path) -> None:
    """Write atomically, so a running app never reads a half-written file."""
    payload = [{"id": c.id, "text": c.page_content, "metadata": c.metadata} for c in chunks]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(partial, path)


def read_snapshot(path: Path) -> list[Document]:
    if not path.exists():
        raise MissingSnapshotError(f"{path} not found; run `make ingest` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Document(id=r["id"], page_content=r["text"], metadata=r["metadata"]) for r in payload]


class LuceneBM25(BM25Okapi):
    """BM25 with Lucene's always-positive IDF.

    `BM25Okapi`'s classic IDF is zero or negative for terms in half or more of the
    documents (then floored to a small epsilon), so common spec words like `server`
    would contribute almost nothing and a match could score below zero.
    """

    def _calc_idf(self, nd: dict[str, int]) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


class BM25Retriever(BaseRetriever):
    """Top `k` documents by BM25 score; documents sharing no terms with the query are dropped."""

    docs: list[Document]
    k: int = 20
    _bm25: LuceneBM25 | None = PrivateAttr(default=None)

    def model_post_init(self, context: object) -> None:
        if self.docs:
            self._bm25 = LuceneBM25([tokenize(d.page_content) for d in self.docs])

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in ranked[: self.k] if scores[i] > 0]


class KeywordIndex:
    """The snapshot's chunks, with one BM25 retriever built lazily per version scope.

    Each scope gets its own retriever so term statistics and the top `k` come from
    in-scope chunks only, rather than filtering a global top `k` after the fact.
    """

    def __init__(self, chunks: list[Document]) -> None:
        self.chunks = chunks
        self.versions: SpecVersions = stored_versions({c.metadata["spec_version"] for c in chunks})
        self._retrievers: dict[tuple[Scope, int], BM25Retriever] = {}

    @classmethod
    def load(cls, cache_dir: Path) -> "KeywordIndex":
        return cls(read_snapshot(cache_dir / SNAPSHOT_NAME))

    def retriever(self, scope: Scope, k: int) -> BM25Retriever:
        key = (scope, k)
        if key not in self._retrievers:
            docs = [c for c in self.chunks if in_scope(c.metadata["spec_version"], scope)]
            self._retrievers[key] = BM25Retriever(docs=docs, k=k)
        return self._retrievers[key]
