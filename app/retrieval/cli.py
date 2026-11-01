"""`uv run query`: run one search and print the ranked chunks, to inspect retrieval."""

import argparse
import sys
import time
from collections.abc import Sequence

from langchain_core.documents import Document

from app.config import MissingSecretError, get_settings
from app.db import make_engine
from app.retrieval.bm25 import MissingSnapshotError
from app.retrieval.scope import ALL, Scope
from app.retrieval.search import Mode, Search

SNIPPET_CHARS = 160


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="query", description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--version", default="latest", help="a dated version, latest or all")
    parser.add_argument("--mode", type=Mode, choices=list(Mode), default=Mode.HYBRID)
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    return parser.parse_args(argv)


def snippet(doc: Document) -> str:
    """Body text without the heading breadcrumb, on one line."""
    body = doc.page_content.removeprefix(doc.metadata.get("heading_path", "")).strip()
    flat = " ".join(body.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 3] + "..."


def describe_scope(version: str, scope: Scope) -> str:
    return ALL if scope is None else f"{version} -> {' + '.join(scope)}"


def format_hit(position: int, doc: Document) -> str:
    meta = doc.metadata
    score = meta.get("relevance_score")
    label = f"{score:.3f}" if score is not None else "  -  "
    lines = [
        f"{position:>2}. {label}  {meta['spec_version']}  {meta['heading_path']}",
        f"           {meta['source_url']}",
    ]
    if "ranks" in meta:
        lines.append("           " + ", ".join(f"{n} #{r}" for n, r in meta["ranks"].items()))
    lines.append(f"           {snippet(doc)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    engine = make_engine(settings.database_url)
    try:
        search = Search.from_settings(settings, engine)
        scope = search.scope(args.version)
        started = time.monotonic()
        hits = search.retriever(args.version, args.mode, args.rerank).invoke(args.question)
    except (MissingSnapshotError, MissingSecretError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    rerank = " + rerank" if args.rerank else ""
    print(
        f"{args.mode}{rerank}, {describe_scope(args.version, scope)}: "
        f"{len(hits)} results in {time.monotonic() - started:.2f}s\n"
    )
    print("\n\n".join(format_hit(i, doc) for i, doc in enumerate(hits, start=1)))
    return 0
