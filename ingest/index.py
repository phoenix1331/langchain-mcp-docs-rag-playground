"""Vector store, record manager and the LangChain indexing call."""

from collections.abc import Callable, Iterable, Iterator

from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import IndexingResult, RecordManager, index
from langchain_core.vectorstores import VectorStore
from langchain_postgres import PGVector
from sqlalchemy import Engine, text

COLLECTION = "mcp_docs"
SOURCE_ID_KEY = "source_path"
BATCH_SIZE = 100


EMBEDDING_TABLE = "langchain_pg_embedding"


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
    engine: Engine, embeddings: Embeddings, dimensions: int, collection: str = COLLECTION
) -> PGVector:
    return PGVector(
        embeddings,
        connection=engine,
        collection_name=collection,
        embedding_length=dimensions,
        use_jsonb=True,
        create_extension=False,  # enabled by scripts/setup-postgres.sh as a superuser
    )


def build_record_manager(engine: Engine, collection: str = COLLECTION) -> SQLRecordManager:
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


def index_chunks(
    chunks: Iterable[Document],
    record_manager: RecordManager,
    vector_store: VectorStore,
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
        key_encoder="sha256",
    )
