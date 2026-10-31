"""Split cleaned pages into retrievable chunks with a heading path breadcrumb.

Pages are split on Markdown headers first, then long sections are split by size.
Each chunk's text starts with its breadcrumb (nav sections › page › headings) so
both BM25 and embeddings see where the text sits in the docs.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import (
    ExperimentalMarkdownSyntaxTextSplitter,
    Language,
    RecursiveCharacterTextSplitter,
)

from ingest.clean_mdx import split_code_fences

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
HEADER_KEYS = tuple(key for _, key in HEADERS)
BREADCRUMB_SEPARATOR = " › "

# Sections with less body text than this (e.g. a lone "**Note:**") are noise.
MIN_BODY_CHARS = 20


def split_sections(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Split on headers, returning `(headings, body)` pairs in page order.

    The experimental splitter keeps whitespace intact but emits each code block as its
    own piece; consecutive pieces under the same headings are merged back together.
    """
    splitter = ExperimentalMarkdownSyntaxTextSplitter(HEADERS, strip_headers=True)
    sections: list[tuple[tuple[str, ...], str]] = []

    for piece in splitter.split_text(text):
        headings = tuple(piece.metadata[k] for k in HEADER_KEYS if k in piece.metadata)
        if sections and sections[-1][0] == headings:
            sections[-1] = (headings, sections[-1][1] + piece.page_content)
            continue
        sections.append((headings, piece.page_content))

    return [(headings, body.strip()) for headings, body in sections]


@dataclass(frozen=True)
class _CodeSpan:
    start: int
    end: int  # end of the closing fence marker, excluding its newline
    opener: str  # e.g. "```typescript"
    closer: str  # e.g. "```"


def _code_spans(text: str) -> list[_CodeSpan]:
    spans: list[_CodeSpan] = []
    offset = 0
    for is_code, segment in split_code_fences(text):
        if is_code:
            opener = segment.splitlines()[0].strip()
            marker = opener[0] * (len(opener) - len(opener.lstrip(opener[0])))
            spans.append(_CodeSpan(offset, offset + len(segment.rstrip("\n")), opener, marker))
        offset += len(segment)
    return spans


def _split_by_size(body: str, splitter: RecursiveCharacterTextSplitter) -> list[str]:
    """Size-split a section, closing and re-opening code fences cut by a split point."""
    spans = _code_spans(body)
    texts: list[str] = []
    for piece in splitter.create_documents([body]):
        text = piece.page_content
        start = piece.metadata["start_index"]
        end = start + len(text)
        for span in spans:
            if span.start < start < span.end:
                text = f"{span.opener}\n{text}"
            if span.start < end < span.end:
                text = f"{text}\n{span.closer}"
        texts.append(text)
    return texts


def heading_path(sections: Iterable[str], page_title: str, headings: Iterable[str]) -> str:
    """Join nav sections, page title and headings, dropping blanks and adjacent repeats."""
    parts: list[str] = []
    for part in (*sections, page_title, *headings):
        label = part.strip()
        if label and (not parts or parts[-1].casefold() != label.casefold()):
            parts.append(label)
    return BREADCRUMB_SEPARATOR.join(parts)


def split_document(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split one page into chunks that carry the page metadata plus `heading_path`."""
    size_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    meta = doc.metadata
    chunks: list[Document] = []

    for headings, body in split_sections(doc.page_content):
        if len(body) < MIN_BODY_CHARS:
            continue
        path = heading_path(meta.get("nav_sections", []), meta.get("page_title", ""), headings)
        for text in _split_by_size(body, size_splitter):
            chunks.append(
                Document(
                    page_content=f"{path}\n\n{text}",
                    metadata={**meta, "heading_path": path},
                )
            )
    return chunks


def split_documents(
    docs: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    return [chunk for doc in docs for chunk in split_document(doc, chunk_size, chunk_overlap)]
