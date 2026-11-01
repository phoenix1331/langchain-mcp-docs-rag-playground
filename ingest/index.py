"""Vector store, record manager and the LangChain indexing call."""

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator

from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import IndexingResult, RecordManager, index
from langchain_core.vectorstores import VectorStore
from langchain_postgres import PGVector
from sqlalchemy import Engine, text

from app.models import slug

COLLECTION_PREFIX = "mcp_docs"
SOURCE_ID_KEY = "source_path"
BATCH_SIZE = 100


EMBEDDING_TABLE = "langchain_pg_embedding"


def collection_name(embed_model: str, dimensions: int) -> str:
    """One collection per embedding model and size.

    The record manager only re-embeds changed text, so switching models in place would
    skip every chunk and leave the old model's vectors behind. A new name forces a full
    index for the new model (repeated text still comes from the embedding cache) and
    leaves the old collection untouched, so switching back is free.
    """
    return f"{COLLECTION_PREFIX}-{slug(embed_model)}-{dimensions}"


class DimensionMismatchError(RuntimeError):
    """The shared embedding column was created with a different vector size."""


def stored_dimensions(engine: Engine) -> int | None:
    """Vector size of the shared embedding column, or None before the table exists."""
    query = text(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = to_regclass(:table) AND attname = 'embedding'"
    )
    with engine.connect() as conn:
        typmod = conn.execute(query, {"table": EMBEDDING_TABLE}).scalar_one_or_none()
    return typmod if typmod and typmod > 0 else None


def check_dimensions(engine: Engine, expected: int) -> None:
    """Fail early: PGVector shares one table across collections and never alters it."""
    actual = stored_dimensions(engine)
    if actual is not None and actual != expected:
        raise DimensionMismatchError(
            f"{EMBEDDING_TABLE}.embedding is vector({actual}) but EMBED_DIMENSIONS is "
            f"{expected}. Drop the langchain_pg_* and upsertion_record tables to rebuild."
        )


def build_vector_store(
    engine: Engine, embeddings: Embeddings, dimensions: int, collection: str
) -> PGVector:
    return PGVector(
        embeddings,
        connection=engine,
        collection_name=collection,
        embedding_length=dimensions,
        use_jsonb=True,
        create_extension=False,  # enabled by scripts/setup-postgres.sh as a superuser
    )


def build_record_manager(engine: Engine, collection: str) -> SQLRecordManager:
    manager = SQLRecordManager(f"pgvector/{collection}", engine=engine)
    manager.create_schema()
    return manager


def _with_progress(
    docs: Iterable[Document], on_progress: Callable[[int], None] | None
) -> Iterator[Document]:
    for count, doc in enumerate(docs, start=1):
        yield doc
        if on_progress and count % BATCH_SIZE == 0:
            on_progress(count)


def collection_key_encoder(collection: str) -> Callable[[Document], str]:
    """Chunk ids hash the collection with the content and metadata.

    PGVector keeps every collection in one table keyed by id alone. With the indexing
    API's default content hash, the same chunk indexed into a second collection would
    upsert the first collection's row instead of adding its own, and cleanup deletes
    (`PGVector.delete` is not collection-scoped by default) would remove other
    collections' rows.
    """

    def encode(doc: Document) -> str:
        payload = json.dumps([collection, doc.page_content, doc.metadata], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    return encode


def index_chunks(
    chunks: Iterable[Document],
    record_manager: RecordManager,
    vector_store: VectorStore,
    collection: str,
    on_progress: Callable[[int], None] | None = None,
) -> IndexingResult:
    """Index the whole corpus. Unchanged chunks are skipped; anything not seen is deleted.

    `full` cleanup (rather than `incremental`) is used because every run indexes the
    whole corpus: it also removes chunks of pages deleted upstream, which incremental
    cleanup never revisits because no documents for those sources arrive.
    """
    return index(
        _with_progress(chunks, on_progress),
        record_manager,
        vector_store,
        batch_size=BATCH_SIZE,
        cleanup="full",
        source_id_key=SOURCE_ID_KEY,
        key_encoder=collection_key_encoder(collection),
    )
