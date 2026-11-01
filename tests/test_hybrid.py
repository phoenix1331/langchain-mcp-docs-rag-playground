from langchain_core.documents import Document

from app.retrieval.hybrid import RRF_K, HybridRetriever, reciprocal_rank_fusion
from tests.retrieval_fakes import StaticRetriever, doc


def test_fusion_rewards_documents_both_retrievers_found() -> None:
    a, b, c = doc("a"), doc("b"), doc("c")

    fused = reciprocal_rank_fusion({"bm25": [a, b], "vector": [c, b]})

    assert [d.id for d in fused] == ["b", "a", "c"]
    assert fused[0].metadata["ranks"] == {"bm25": 2, "vector": 2}
    assert fused[0].metadata["rrf_score"] == 2 / (RRF_K + 2)
    assert fused[1].metadata["ranks"] == {"bm25": 1}


def test_fusion_keys_on_id_so_identical_text_in_two_versions_survives() -> None:
    old = Document(id="old", page_content="same", metadata={"spec_version": "2025-11-25"})
    new = Document(id="new", page_content="same", metadata={"spec_version": "2026-07-28"})

    fused = reciprocal_rank_fusion({"bm25": [old, new], "vector": [new]})

    assert [d.id for d in fused] == ["new", "old"]


def test_fusion_falls_back_to_content_without_ids() -> None:
    fused = reciprocal_rank_fusion(
        {"bm25": [Document(page_content="x")], "vector": [Document(page_content="x")]}
    )

    assert len(fused) == 1


def test_fusion_does_not_mutate_inputs() -> None:
    original = doc("a")

    reciprocal_rank_fusion({"bm25": [original]})

    assert "ranks" not in original.metadata


def test_hybrid_retriever_fuses_and_truncates() -> None:
    retriever = HybridRetriever(
        retrievers={
            "bm25": StaticRetriever(docs=[doc("a"), doc("b"), doc("c")]),
            "vector": StaticRetriever(docs=[doc("c"), doc("d")]),
        },
        k=2,
    )

    assert [d.id for d in retriever.invoke("q")] == ["c", "a"]
