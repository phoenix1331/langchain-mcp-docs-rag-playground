"""Voyage reranking of the candidate chunks down to `RERANK_TOP_N`; built by `app.models`."""

from collections.abc import Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_voyageai import VoyageAIRerank
from voyageai.object.reranking import RerankingObject


def _ranked(documents: Sequence[Document], result: RerankingObject) -> list[Document]:
    return [
        documents[r.index].model_copy(
            update={
                "metadata": {**documents[r.index].metadata, "relevance_score": r.relevance_score}
            }
        )
        for r in result.results
    ]


class VoyageReranker(VoyageAIRerank):
    """`VoyageAIRerank` that keeps each document's id, which the upstream copy drops.

    Ids matter downstream: citations and evals match chunks by id.
    """

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> Sequence[Document]:
        if not documents:
            return []
        return _ranked(documents, self._rerank(documents, query))

    async def acompress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> Sequence[Document]:
        if not documents:
            return []
        return _ranked(documents, await self._arerank(documents, query))
