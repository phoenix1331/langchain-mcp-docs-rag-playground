"""Model factories. Every model setting is `provider:model`, so providers can be swapped.

Chat models go through LangChain's `init_chat_model` (`google_genai`, `anthropic`,
`openai`, `ollama`, ...) and embeddings through `init_embeddings`. LangChain has no
factory entry for Voyage embeddings and none at all for rerankers, so those are mapped
here. Each integration reads its own API key from the environment (`GOOGLE_API_KEY`,
`VOYAGE_API_KEY`, `OPENAI_API_KEY`, ...); install the integration package for any
provider you switch to.
"""

import re
from collections.abc import Callable

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_voyageai import VoyageAIEmbeddings

from app.config import MODEL_SPEC
from app.retrieval.rerank import VoyageReranker

VOYAGE = "voyageai"


class ModelConfigError(ValueError):
    """A model setting is malformed or names a provider this project cannot build."""


def parse_spec(spec: str) -> tuple[str, str]:
    """Split `provider:model`; the model part may itself contain colons."""
    if not re.match(MODEL_SPEC, spec):
        raise ModelConfigError(
            f"{spec!r} must be provider:model, e.g. google_genai:gemini-3.6-flash"
        )
    provider, _, name = spec.partition(":")
    return provider, name


def slug(spec: str) -> str:
    """Filesystem- and key-safe form of a spec: `voyageai:voyage-4` -> `voyageai-voyage-4`."""
    return re.sub(r"[^a-z0-9.]+", "-", spec.lower()).strip("-")


def chat_model(spec: str) -> BaseChatModel:
    parse_spec(spec)
    return init_chat_model(spec, temperature=0)


def embeddings(spec: str, dimensions: int) -> Embeddings:
    """Voyage gets `dimensions` as its output size; other providers use their default.

    `EMBED_DIMENSIONS` must match whatever the chosen model returns.
    """
    provider, name = parse_spec(spec)
    if provider == VOYAGE:
        # Batching is done by the ingest's RateLimitedEmbeddings, so the client's is lifted.
        return VoyageAIEmbeddings(model=name, output_dimension=dimensions, batch_size=1000)
    return init_embeddings(spec)


def _voyage_reranker(name: str, top_n: int) -> BaseDocumentCompressor:
    return VoyageReranker(model=name, top_k=top_n)


def _cohere_reranker(name: str, top_n: int) -> BaseDocumentCompressor:
    try:
        from langchain_cohere import CohereRerank
    except ImportError as exc:
        raise ModelConfigError("cohere reranking needs `uv add langchain-cohere`") from exc
    return CohereRerank(model=name, top_n=top_n)


RERANKERS: dict[str, Callable[[str, int], BaseDocumentCompressor]] = {
    VOYAGE: _voyage_reranker,
    "cohere": _cohere_reranker,
}


def reranker(spec: str, top_n: int) -> BaseDocumentCompressor:
    """Any `BaseDocumentCompressor` works downstream; add a provider by adding a factory."""
    provider, name = parse_spec(spec)
    factory = RERANKERS.get(provider)
    if factory is None:
        raise ModelConfigError(
            f"no reranker for provider {provider!r}; supported: {', '.join(RERANKERS)}"
        )
    return factory(name, top_n)
