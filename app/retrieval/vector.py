"""Semantic search over pgvector, filtered to a version scope."""

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_postgres import PGVector
from sqlalchemy import Engine

from app.config import Settings
from app.models import embeddings as embedding_model
from app.retrieval.scope import Scope
from ingest.index import build_vector_store, collection_name


def scope_filter(scope: Scope) -> dict[str, Any] | None:
    """PGVector metadata filter for a scope; None means no filter."""
    if scope is None:
        return None
    return {"spec_version": {"$in": list(scope)}}


def query_vector_store(
    settings: Settings, engine: Engine, embeddings: Embeddings | None = None
) -> PGVector:
    """The ingest's collection for `EMBED_MODEL`, embedding queries without the rate limiter."""
    return build_vector_store(
        engine,
        embeddings or embedding_model(settings.embed_model, settings.embed_dimensions),
        settings.embed_dimensions,
        collection_name(settings.embed_model, settings.embed_dimensions),
    )


def vector_retriever(store: VectorStore, scope: Scope, k: int) -> VectorStoreRetriever:
    return store.as_retriever(search_kwargs={"k": k, "filter": scope_filter(scope)})
