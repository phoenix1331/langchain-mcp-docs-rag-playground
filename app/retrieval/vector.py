"""Semantic search over pgvector, filtered to a version scope."""

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_postgres import PGVector
from sqlalchemy import Engine

from app.config import Settings
from app.retrieval.scope import Scope
from ingest.embeddings import voyage_embeddings
from ingest.index import build_vector_store


def scope_filter(scope: Scope) -> dict[str, Any] | None:
    """PGVector metadata filter for a scope; None means no filter."""
    if scope is None:
        return None
    return {"spec_version": {"$in": list(scope)}}


def query_vector_store(
    settings: Settings, engine: Engine, embeddings: Embeddings | None = None
) -> PGVector:
    """The ingest's vector store, embedding queries with Voyage (no ingest rate limiter)."""
    return build_vector_store(
        engine, embeddings or voyage_embeddings(settings), settings.embed_dimensions
    )


def vector_retriever(store: VectorStore, scope: Scope, k: int) -> VectorStoreRetriever:
    return store.as_retriever(search_kwargs={"k": k, "filter": scope_filter(scope)})
