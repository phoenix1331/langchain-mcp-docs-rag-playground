from pathlib import Path

import pytest

from ingest.fetch import SpecVersions
from ingest.load import iter_source_files, load_document, load_documents

VERSIONS = SpecVersions(versions=("2025-06-18", "2026-07-28"))


def write(repo: Path, rel: str, text: str = "---\ntitle: Page\n---\nBody\n") -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for rel in [
        "docs/specification/2025-06-18/basic/lifecycle.mdx",
        "docs/specification/2026-07-28/index.mdx",
        "docs/specification/draft/index.mdx",
        "docs/docs/2026-07-28/learn/architecture.mdx",
        "docs/docs/draft/learn/architecture.mdx",
        "docs/extensions/overview.mdx",
        "docs/registry/about.mdx",
        "docs/seps/1.mdx",
        "docs/community/governance.mdx",
        "docs/examples.mdx",
    ]:
        write(tmp_path, rel)
    write(tmp_path, "docs/specification/2026-07-28/notes.md")
    return tmp_path


def test_iter_source_files_selects_scope(repo: Path) -> None:
    found = [(path.as_posix(), version) for path, version in iter_source_files(repo, VERSIONS)]

    assert found == [
        ("docs/specification/2025-06-18/basic/lifecycle.mdx", "2025-06-18"),
        ("docs/specification/2026-07-28/index.mdx", "2026-07-28"),
        ("docs/docs/2026-07-28/learn/architecture.mdx", "2026-07-28"),
        ("docs/extensions/overview.mdx", "unversioned"),
        ("docs/examples.mdx", "unversioned"),
    ]


def test_iter_source_files_tolerates_missing_optional_paths(tmp_path: Path) -> None:
    write(tmp_path, "docs/specification/2026-07-28/index.mdx")

    found = list(iter_source_files(tmp_path, SpecVersions(versions=("2026-07-28",))))

    assert [path.as_posix() for path, _ in found] == ["docs/specification/2026-07-28/index.mdx"]


def test_load_document_sets_metadata(tmp_path: Path) -> None:
    write(
        tmp_path,
        "docs/specification/2026-07-28/basic/index.mdx",
        "---\ntitle: Overview\ndescription: Basics\n---\n<Note>Hi</Note>\n",
    )

    path = Path("docs/specification/2026-07-28/basic/index.mdx")

    doc = load_document(tmp_path, path, "2026-07-28")

    assert doc.metadata == {
        "source_path": "docs/specification/2026-07-28/basic/index.mdx",
        "spec_version": "2026-07-28",
        "page_title": "Overview",
        "description": "Basics",
    }
    assert "**Note:** Hi" in doc.page_content


def test_load_document_falls_back_to_file_name(tmp_path: Path) -> None:
    write(tmp_path, "docs/extensions/apps.mdx", "No frontmatter\n")

    doc = load_document(tmp_path, Path("docs/extensions/apps.mdx"), "unversioned")

    assert doc.metadata["page_title"] == "apps"
    assert doc.metadata["description"] == ""


def test_load_documents_skips_pages_empty_after_cleaning(repo: Path) -> None:
    write(repo, "docs/extensions/empty.mdx", "---\ntitle: Empty\n---\n<Icon icon='x' />\n")

    docs = load_documents(repo, VERSIONS)

    assert len(docs) == 5
    assert "docs/extensions/empty.mdx" not in {d.metadata["source_path"] for d in docs}
