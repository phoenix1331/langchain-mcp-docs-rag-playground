import sys
from types import ModuleType, SimpleNamespace

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_voyageai import VoyageAIEmbeddings

from app import models
from app.models import (
    REQUEST_TIMEOUT_SECONDS,
    ModelConfigError,
    chat_model,
    embeddings,
    parse_spec,
    reranker,
    slug,
)
from app.retrieval.rerank import VoyageReranker


@pytest.fixture(autouse=True)
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setenv("VOYAGE_API_KEY", "key")


def test_parse_spec_keeps_colons_in_the_model() -> None:
    assert parse_spec("google_genai:gemini-3.6-flash") == ("google_genai", "gemini-3.6-flash")
    assert parse_spec("ollama:llama3.2:3b") == ("ollama", "llama3.2:3b")


@pytest.mark.parametrize("spec", ["gemini", "voyageai:", ":model", "Voyage:x"])
def test_parse_spec_rejects_malformed(spec: str) -> None:
    with pytest.raises(ModelConfigError, match="must be provider:model"):
        parse_spec(spec)


def test_slug() -> None:
    assert slug("voyageai:voyage-4") == "voyageai-voyage-4"
    assert slug("ollama:Llama3.2:3B") == "ollama-llama3.2-3b"


def test_chat_model_goes_through_init_chat_model() -> None:
    llm = chat_model("google_genai:gemini-test")

    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model.endswith("gemini-test")
    assert llm.temperature == 0
    assert llm.timeout == REQUEST_TIMEOUT_SECONDS


def test_voyage_embeddings_get_the_configured_size() -> None:
    built = embeddings("voyageai:voyage-test", 512)

    assert isinstance(built, VoyageAIEmbeddings)
    assert (built.model, built.output_dimension) == ("voyage-test", 512)


def test_other_embedding_providers_use_init_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "init_embeddings", lambda spec: f"built {spec}")

    spec = "openai:text-embedding-3-small"

    assert embeddings(spec, 1536) == f"built {spec}"


def test_voyage_reranker() -> None:
    built = reranker("voyageai:rerank-test", 4)

    assert isinstance(built, VoyageReranker)
    assert (built.model, built.top_k) == ("rerank-test", 4)


def test_cohere_reranker_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("langchain_cohere")
    module.CohereRerank = lambda model, top_n: SimpleNamespace(model=model, top_n=top_n)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_cohere", module)

    built = reranker("cohere:rerank-v3.5", 3)

    assert (built.model, built.top_n) == ("rerank-v3.5", 3)  # type: ignore[attr-defined]


def test_cohere_reranker_explains_the_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langchain_cohere", None)

    with pytest.raises(ModelConfigError, match="uv add langchain-cohere"):
        reranker("cohere:rerank-v3.5", 3)


def test_unknown_reranker_provider_lists_the_supported_ones() -> None:
    with pytest.raises(ModelConfigError, match="supported: voyageai, cohere"):
        reranker("openai:gpt-rerank", 3)
