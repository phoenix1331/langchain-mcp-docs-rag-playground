"""Embedding wrappers: client-side rate limiting and a content-hash cache.

Voyage without a payment method allows 3 requests and 10K tokens per minute. The
rate limiter keeps ingestion under configured limits instead of failing on 429s;
the cache means text repeated across spec versions (or re-runs) is embedded once.
"""

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import SecretStr

from app.config import Settings, require_secret

WINDOW_SECONDS = 60.0
CHARS_PER_TOKEN = 4  # conservative: Voyage counts roughly 16% fewer tokens than this


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


class RateLimiter:
    """Sliding one-minute window over requests and tokens. A limit of 0 means unlimited."""

    def __init__(
        self,
        rpm: int,
        tpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()

    def acquire(self, tokens: int) -> None:
        """Block until a request of `tokens` fits in the window, then record it."""
        while True:
            now = self._clock()
            while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
                self._events.popleft()
            if self._fits(tokens):
                self._events.append((now, tokens))
                return
            self._sleep(max(self._events[0][0] + WINDOW_SECONDS - now, 0.01))

    def _fits(self, tokens: int) -> bool:
        if not self._events:
            return True  # an oversized single request must still be allowed through
        if self.rpm and len(self._events) >= self.rpm:
            return False
        used = sum(t for _, t in self._events)
        return not self.tpm or used + tokens <= self.tpm


def batch_by_tokens(texts: list[str], max_tokens: int) -> list[list[str]]:
    """Group texts into batches whose estimated tokens stay within `max_tokens`."""
    batches: list[list[str]] = []
    current: list[str] = []
    used = 0
    for text in texts:
        tokens = estimate_tokens(text)
        if current and used + tokens > max_tokens:
            batches.append(current)
            current, used = [], 0
        current.append(text)
        used += tokens
    if current:
        batches.append(current)
    return batches


class RateLimitedEmbeddings(Embeddings):
    """Sends token-bounded batches to `inner`, waiting on `limiter` before each request."""

    def __init__(
        self,
        inner: Embeddings,
        limiter: RateLimiter,
        batch_tokens: int,
        on_batch: Callable[[int], None] | None = None,
    ) -> None:
        self.inner = inner
        self.limiter = limiter
        self.batch_tokens = batch_tokens
        self.on_batch = on_batch

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each distinct text once; duplicates in the same call share a vector."""
        unique = list(dict.fromkeys(texts))
        vectors: dict[str, list[float]] = {}
        for batch in batch_by_tokens(unique, self.batch_tokens):
            self.limiter.acquire(sum(estimate_tokens(t) for t in batch))
            vectors.update(zip(batch, self.inner.embed_documents(batch), strict=True))
            if self.on_batch:
                self.on_batch(len(batch))
        return [vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.limiter.acquire(estimate_tokens(text))
        return self.inner.embed_query(text)


def voyage_embeddings(settings: Settings) -> VoyageAIEmbeddings:
    return VoyageAIEmbeddings(
        model=settings.embed_model,
        voyage_api_key=SecretStr(require_secret(settings.voyage_api_key, "VOYAGE_API_KEY")),
        output_dimension=settings.embed_dimensions,
        batch_size=1000,  # batching is done by RateLimitedEmbeddings
    )


def cache_namespace(settings: Settings) -> str:
    """Cache keys include the model and dimensions, so changing either re-embeds."""
    return f"{settings.embed_model}-{settings.embed_dimensions}"


def build_ingest_embeddings(
    settings: Settings,
    inner: Embeddings | None = None,
    on_batch: Callable[[int], None] | None = None,
) -> CacheBackedEmbeddings:
    """Cache(RateLimited(Voyage)): only cache misses reach the API, within rate limits."""
    limited = RateLimitedEmbeddings(
        inner or voyage_embeddings(settings),
        RateLimiter(settings.embed_rpm, settings.embed_tpm),
        settings.embed_batch_tokens,
        on_batch,
    )
    store = LocalFileStore(Path(settings.cache_dir) / "embeddings")
    return CacheBackedEmbeddings.from_bytes_store(
        limited, store, namespace=cache_namespace(settings), key_encoder="sha256"
    )
