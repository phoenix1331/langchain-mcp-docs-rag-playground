import asyncio
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from pydantic import SecretStr

from app.config import MissingSecretError, Settings
from app.retrieval.rerank import VoyageReranker, build_reranker
from tests.retrieval_fakes import doc


def fake_result(documents: Sequence[Document | str], query: str) -> SimpleNamespace:
    # Voyage returns the best first; here the last candidate wins.
    return SimpleNamespace(
        results=[
            SimpleNamespace(index=2, relevance_score=0.9),
            SimpleNamespace(index=0, relevance_score=0.4),
        ]
    )


@pytest.fixture
def reranker(monkeypatch: pytest.MonkeyPatch) -> VoyageReranker:
    async def afake(
        self: VoyageReranker, documents: Sequence[Document], query: str
    ) -> SimpleNamespace:
        return fake_result(documents, query)

    monkeypatch.setattr(VoyageReranker, "_rerank", lambda self, d, q: fake_result(d, q))
    monkeypatch.setattr(VoyageReranker, "_arerank", afake)
    return VoyageReranker(model="rerank-test", top_k=2, voyage_api_key=SecretStr("key"))


def test_rerank_orders_scores_and_keeps_ids(reranker: VoyageReranker) -> None:
    candidates = [doc("a"), doc("b"), doc("c")]

    ranked = reranker.compress_documents(candidates, "q")

    assert [(d.id, d.metadata["relevance_score"]) for d in ranked] == [("c", 0.9), ("a", 0.4)]
    assert "relevance_score" not in candidates[2].metadata


def test_async_rerank_keeps_ids(reranker: VoyageReranker) -> None:
    ranked = asyncio.run(reranker.acompress_documents([doc("a"), doc("b"), doc("c")], "q"))

    assert [d.id for d in ranked] == ["c", "a"]


def test_empty_candidates_skip_the_api(reranker: VoyageReranker) -> None:
    assert reranker.compress_documents([], "q") == []
    assert asyncio.run(reranker.acompress_documents([], "q")) == []


def settings(tmp_path: Path, key: str | None) -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql+psycopg://u:p@localhost/db",
        router_model="r",
        answer_model="a",
        embed_model="e",
        rerank_model="rerank-test",
        rerank_top_n=4,
        voyage_api_key=SecretStr(key) if key else None,
        cache_dir=tmp_path,
    )


def test_build_reranker_uses_config(tmp_path: Path) -> None:
    built = build_reranker(settings(tmp_path, "key"))

    assert (built.model, built.top_k) == ("rerank-test", 4)


def test_build_reranker_needs_a_key(tmp_path: Path) -> None:
    with pytest.raises(MissingSecretError, match="VOYAGE_API_KEY"):
        build_reranker(settings(tmp_path, None))
