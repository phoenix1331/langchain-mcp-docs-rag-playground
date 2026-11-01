"""Shared stand-ins for retrieval tests."""

from collections.abc import Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun, Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.retrievers import BaseRetriever


def doc(key: str, version: str = "2026-07-28", heading: str = "Base › Page") -> Document:
    return Document(
        id=key,
        page_content=f"{heading}\n\nbody of {key}",
        metadata={
            "spec_version": version,
            "heading_path": heading,
            "source_url": f"https://example.test/{key}",
            "source_path": f"docs/{key}.mdx",
        },
    )


class StaticRetriever(BaseRetriever):
    """Returns the same documents for every query."""

    docs: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.docs


class ReverseReranker(BaseDocumentCompressor):
    """Reverses the candidates, keeps `top_n` and scores them 1.0, 0.9, ..."""

    top_n: int = 2

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> Sequence[Document]:
        picked = list(reversed(documents))[: self.top_n]
        return [
            d.model_copy(update={"metadata": {**d.metadata, "relevance_score": 1 - i / 10}})
            for i, d in enumerate(picked)
        ]
