from textwrap import dedent

from langchain_core.documents import Document

from ingest.clean_mdx import split_code_fences
from ingest.split import heading_path, split_document, split_documents, split_sections

META = {
    "source_path": "docs/specification/2026-07-28/basic/authorization/index.mdx",
    "spec_version": "2026-07-28",
    "page_title": "Authorization",
    "source_url": "https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization",
    "nav_tab": "Specification",
    "nav_sections": ["Base Protocol", "Authorization"],
}


def md(text: str) -> str:
    return dedent(text).lstrip("\n")


def page(text: str, **meta: object) -> Document:
    return Document(page_content=md(text), metadata={**META, **meta})


def fences_balanced(text: str) -> bool:
    segments = split_code_fences(text)
    return not segments or not segments[-1][0] or segments[-1][1].rstrip().endswith("```")


class TestSplitSections:
    def test_code_blocks_stay_with_their_section(self) -> None:
        text = md("""
            ## Setup

            Install it:

            ```bash
            # not a header
            pip install mcp
            ```

            Then run it.
        """)

        sections = split_sections(text)

        assert len(sections) == 1
        headings, body = sections[0]
        assert headings == ("Setup",)
        assert body.startswith("Install it:")
        assert "# not a header" in body
        assert body.endswith("Then run it.")

    def test_nested_headings_and_header_only_sections_are_omitted(self) -> None:
        text = "# Title\n\nIntro\n\n## Empty\n\n## Tokens\n\n### Refresh\n\nRotate them.\n"

        sections = split_sections(text)

        assert sections == [
            (("Title",), "Intro"),
            (("Title", "Tokens", "Refresh"), "Rotate them."),
        ]


class TestHeadingPath:
    def test_joins_sections_title_and_headings(self) -> None:
        path = heading_path(["Base Protocol"], "Lifecycle", ["Operation", "Shutdown"])

        assert path == "Base Protocol › Lifecycle › Operation › Shutdown"

    def test_drops_blanks_and_adjacent_repeats(self) -> None:
        path = heading_path(["Base Protocol", "Authorization"], "authorization", ["", "Tokens"])

        assert path == "Base Protocol › Authorization › Tokens"


class TestSplitDocument:
    def test_chunks_carry_metadata_and_breadcrumb(self) -> None:
        doc = page("""
            ## Token Handling

            Clients must send the access token in the Authorization header.
        """)

        [chunk] = split_document(doc, chunk_size=500, chunk_overlap=0)

        expected_path = "Base Protocol › Authorization › Token Handling"
        assert chunk.metadata == {**META, "heading_path": expected_path}
        assert chunk.page_content == (
            f"{expected_path}\n\nClients must send the access token in the Authorization header."
        )

    def test_short_sections_are_skipped(self) -> None:
        doc = page("""
            ## Notes

            **Note:**

            ## Real

            This section has enough text to keep.
        """)

        chunks = split_document(doc, chunk_size=500, chunk_overlap=0)

        assert [c.metadata["heading_path"].split(" › ")[-1] for c in chunks] == ["Real"]

    def test_long_sections_are_split_by_size(self) -> None:
        paragraphs = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(6))
        doc = page(f"## Long\n\n{paragraphs}\n")

        chunks = split_document(doc, chunk_size=400, chunk_overlap=0)

        assert len(chunks) > 1
        prefix = "Base Protocol › Authorization › Long\n\n"
        assert all(c.page_content.startswith(prefix) for c in chunks)

    def test_code_blocks_cut_by_size_are_closed_and_reopened(self) -> None:
        code = "\n".join(f"const value{i} = compute({i});" for i in range(40))
        doc = page(f"## Example\n\nSee below.\n\n```typescript\n{code}\n```\n\nAfter the code.\n")

        chunks = split_document(doc, chunk_size=300, chunk_overlap=60)

        assert len(chunks) > 2
        assert all(fences_balanced(c.page_content) for c in chunks)
        assert all(c.page_content.count("```") % 2 == 0 for c in chunks)
        middle = chunks[1].page_content.split("\n\n", 1)[1]
        assert middle.startswith("```typescript\n")
        assert chunks[-1].page_content.endswith("After the code.")

    def test_page_without_nav_sections(self) -> None:
        doc = Document(page_content="Plain text long enough to keep.", metadata={"page_title": "X"})

        [chunk] = split_document(doc, chunk_size=500, chunk_overlap=0)

        assert chunk.metadata["heading_path"] == "X"


def test_split_documents_flattens_pages() -> None:
    docs = [page("## A\n\nFirst section body text.\n"), page("## B\n\nSecond section body text.\n")]

    chunks = split_documents(docs, chunk_size=500, chunk_overlap=0)

    assert [c.metadata["heading_path"].split(" › ")[-1] for c in chunks] == ["A", "B"]
