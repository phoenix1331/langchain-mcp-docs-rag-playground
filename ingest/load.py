"""Select in-scope MDX files from the MCP repo and load them as cleaned Documents."""

from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document

from ingest.clean_mdx import clean_mdx
from ingest.fetch import DOCS_DIR, GUIDES_DIR, SPEC_DIR, UNVERSIONED, SpecVersions
from ingest.navigation import Navigation

# Unversioned areas in scope. Registry, SEPs and community pages are excluded.
UNVERSIONED_DIRS = (DOCS_DIR / "extensions",)
UNVERSIONED_FILES = (DOCS_DIR / "examples.mdx",)


def iter_source_files(repo: Path, versions: SpecVersions) -> Iterator[tuple[Path, str]]:
    """Yield `(path relative to repo, spec_version)` for every in-scope MDX file, sorted."""
    for version in versions.versions:
        for base in (SPEC_DIR, GUIDES_DIR):
            for path in sorted((repo / base / version).rglob("*.mdx")):
                yield path.relative_to(repo), version

    for base in UNVERSIONED_DIRS:
        for path in sorted((repo / base).rglob("*.mdx")):
            yield path.relative_to(repo), UNVERSIONED

    for path in UNVERSIONED_FILES:
        if (repo / path).is_file():
            yield path, UNVERSIONED


def load_document(
    repo: Path, source_path: Path, spec_version: str, navigation: Navigation
) -> Document:
    """Read and clean one MDX file. `page_title` falls back to the file name."""
    cleaned = clean_mdx((repo / source_path).read_text(encoding="utf-8"))
    front = cleaned.metadata
    nav = navigation.lookup(source_path)
    return Document(
        page_content=cleaned.content,
        metadata={
            "source_path": source_path.as_posix(),
            "spec_version": spec_version,
            "page_title": str(front.get("title") or source_path.stem),
            "description": str(front.get("description") or ""),
            "source_url": nav.url,
            "nav_tab": nav.tab,
            "nav_sections": list(nav.sections),
        },
    )


def load_documents(repo: Path, versions: SpecVersions) -> list[Document]:
    """Load every in-scope page, skipping pages that are empty after cleaning."""
    navigation = Navigation.from_repo(repo)
    documents = (
        load_document(repo, path, version, navigation)
        for path, version in iter_source_files(repo, versions)
    )
    return [doc for doc in documents if doc.page_content.strip()]
