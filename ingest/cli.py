"""`uv run ingest`: fetch the MCP docs, chunk them and index incrementally into pgvector."""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from app.config import MissingSecretError, Settings, get_settings
from app.db import make_engine
from ingest.embeddings import build_ingest_embeddings
from ingest.fetch import FetchError, discover_versions, sync_repo
from ingest.index import (
    DimensionMismatchError,
    build_record_manager,
    build_vector_store,
    check_dimensions,
    index_chunks,
)
from ingest.load import load_documents
from ingest.split import split_documents

REPO_DIR_NAME = "modelcontextprotocol"


@dataclass(frozen=True)
class IngestSummary:
    commit: str
    versions: tuple[str, ...]
    pages: int
    chunks: int
    added: int
    updated: int
    skipped: int
    deleted: int
    embedded: int
    seconds: float


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run_ingest(
    settings: Settings,
    inner_embeddings: Embeddings | None = None,
    log: Callable[[str], None] = _log,
) -> IngestSummary:
    started = time.monotonic()
    repo = settings.cache_dir / REPO_DIR_NAME

    log(f"syncing {settings.mcp_repo_url} ...")
    commit = sync_repo(settings.mcp_repo_url, repo)
    versions = discover_versions(repo)
    log(f"commit {commit[:12]}; spec versions {', '.join(versions.versions)}")

    pages = load_documents(repo, versions)
    chunks = split_documents(pages, settings.chunk_size, settings.chunk_overlap)
    log(f"{len(pages)} pages -> {len(chunks)} chunks")

    embedded = 0

    def on_batch(count: int) -> None:
        nonlocal embedded
        embedded += count
        log(f"  embedded {embedded} new texts")

    def on_retry(attempt: int) -> None:
        log(f"  rate limited by the API; waiting a minute (retry {attempt})")

    embeddings = build_ingest_embeddings(settings, inner_embeddings, on_batch, on_retry)
    engine = make_engine(settings.database_url)
    try:
        check_dimensions(engine, settings.embed_dimensions)
        result = index_chunks(
            chunks,
            build_record_manager(engine),
            build_vector_store(engine, embeddings, settings.embed_dimensions),
            on_progress=lambda n: log(f"indexed {n}/{len(chunks)} chunks"),
        )
    finally:
        engine.dispose()

    return IngestSummary(
        commit=commit,
        versions=versions.versions,
        pages=len(pages),
        chunks=len(chunks),
        added=result["num_added"],
        updated=result["num_updated"],
        skipped=result["num_skipped"],
        deleted=result["num_deleted"],
        embedded=embedded,
        seconds=time.monotonic() - started,
    )


def format_summary(summary: IngestSummary) -> str:
    return "\n".join(
        [
            f"commit:   {summary.commit[:12]} (latest spec {summary.versions[-1]})",
            f"pages:    {summary.pages}",
            f"chunks:   {summary.chunks}",
            f"added:    {summary.added}",
            f"updated:  {summary.updated}",
            f"skipped:  {summary.skipped}",
            f"deleted:  {summary.deleted}",
            f"embedded: {summary.embedded} (cache misses sent to the API)",
            f"took:     {summary.seconds:.1f}s",
        ]
    )


def main() -> int:
    try:
        summary = run_ingest(get_settings())
    except (FetchError, MissingSecretError, DimensionMismatchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_summary(summary))
    return 0
