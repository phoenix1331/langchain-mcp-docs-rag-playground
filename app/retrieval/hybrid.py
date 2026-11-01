"""Hybrid search: BM25 and vector results merged with reciprocal rank fusion.

LangChain's `EnsembleRetriever` does RRF too, but it de-duplicates on `page_content`
(or a metadata key). The same section appears word for word in several spec versions,
so under the `all` scope it would merge chunks from different versions into one. This
retriever fuses on the stored chunk id instead and records where each hit came from.
"""

from collections.abc import Mapping, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

RRF_K = 60  # the standard constant from the original RRF paper; damps top-rank dominance


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Document]], c: int = RRF_K
) -> list[Document]:
    """Order documents by summed `1 / (c + rank)` across rankings, best first.

    Each result is a copy carrying `ranks` ({retriever name: 1-based rank}) and
    `rrf_score` in its metadata; inputs are never mutated (BM25 documents are shared).
    Ties keep first-seen order.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    first: dict[str, Document] = {}
    for name, ranking in rankings.items():
        for rank, doc in enumerate(ranking, start=1):
            key = doc.id or doc.page_content
            scores[key] = scores.get(key, 0.0) + 1 / (c + rank)
            ranks.setdefault(key, {})[name] = rank
            first.setdefault(key, doc)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return [
        first[key].model_copy(update={"metadata": _annotated(first[key], ranks[key], scores[key])})
        for key in ordered
    ]


def _annotated(doc: Document, ranks: dict[str, int], score: float) -> dict[str, object]:
    return {**doc.metadata, "ranks": ranks, "rrf_score": score}


class HybridRetriever(BaseRetriever):
    """Runs each named retriever, fuses their rankings and keeps the top `k`."""

    retrievers: dict[str, BaseRetriever]
    k: int = 20

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        rankings = {
            name: retriever.invoke(query, config={"callbacks": run_manager.get_child(name)})
            for name, retriever in self.retrievers.items()
        }
        return reciprocal_rank_fusion(rankings)[: self.k]
