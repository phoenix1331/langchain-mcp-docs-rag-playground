"""Builds the retriever for a question: vector, BM25 or hybrid, optionally reranked."""

from dataclasses import dataclass
from enum import StrEnum

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from sqlalchemy import Engine

from app.config import Settings
from app.retrieval.bm25 import KeywordIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rerank import build_reranker
from app.retrieval.scope import Scope, version_scope
from app.retrieval.vector import query_vector_store, vector_retriever


class Mode(StrEnum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class Search:
    """Long-lived search resources; `retriever()` is cheap and called per question."""

    keyword: KeywordIndex
    store: VectorStore
    reranker: BaseDocumentCompressor
    top_k: int

    @classmethod
    def from_settings(cls, settings: Settings, engine: Engine) -> "Search":
        return cls(
            keyword=KeywordIndex.load(settings.cache_dir),
            store=query_vector_store(settings, engine),
            reranker=build_reranker(settings),
            top_k=settings.top_k,
        )

    def scope(self, version: str) -> Scope:
        return version_scope(version, self.keyword.versions)

    def retriever(
        self, version: str, mode: Mode = Mode.HYBRID, rerank: bool = True
    ) -> BaseRetriever:
        """`top_k` candidates from the chosen mode, reranked to `RERANK_TOP_N` if `rerank`."""
        scope = self.scope(version)
        candidates: dict[str, BaseRetriever] = {
            Mode.BM25: self.keyword.retriever(scope, self.top_k),
            Mode.VECTOR: vector_retriever(self.store, scope, self.top_k),
        }
        base = candidates.get(mode) or HybridRetriever(retrievers=candidates, k=self.top_k)
        if not rerank:
            return base
        return ContextualCompressionRetriever(base_compressor=self.reranker, base_retriever=base)
