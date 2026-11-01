"""Embedding wrappers: client-side rate limiting and a content-hash cache.

Free tiers (Voyage without a payment method: 3 requests and 10K tokens per minute) are
tight. The rate limiter keeps ingestion under configured limits instead of failing on 429s;
the cache means text repeated across spec versions (or re-runs) is embedded once.
"""

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import voyageai
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from voyageai.error import RateLimitError

from app.config import Settings
from app.models import VOYAGE, embeddings, parse_spec, slug

WINDOW_SECONDS = 60.0
CHARS_PER_TOKEN = 4  # rough fallback only: code-heavy text can run 2.5x higher
MAX_RETRIES = 5

TokenCounter = Callable[[str], int]
T = TypeVar("T")


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


def batch_by_tokens(
    texts: list[str], max_tokens: int, count: TokenCounter = estimate_tokens
) -> list[tuple[list[str], int]]:
    """Group texts into `(batch, tokens)` pairs whose token counts stay within `max_tokens`."""
    batches: list[tuple[list[str], int]] = []
    current: list[str] = []
    used = 0
    for text in texts:
        tokens = count(text)
        if current and used + tokens > max_tokens:
            batches.append((current, used))
            current, used = [], 0
        current.append(text)
        used += tokens
    if current:
        batches.append((current, used))
    return batches


class RateLimitedEmbeddings(Embeddings):
    """Sends token-bounded batches to `inner`, waiting on `limiter` before each request.

    If the API still rate-limits (estimates off, or another client sharing the key), the
    request is retried after a full window, up to `max_retries` times.
    """

    def __init__(
        self,
        inner: Embeddings,
        limiter: RateLimiter,
        batch_tokens: int,
        on_batch: Callable[[int], None] | None = None,
        count_tokens: TokenCounter = estimate_tokens,
        retry_on: tuple[type[Exception], ...] = (),
        max_retries: int = MAX_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
        on_retry: Callable[[int], None] | None = None,
    ) -> None:
        self.inner = inner
        self.limiter = limiter
        self.batch_tokens = batch_tokens
        self.on_batch = on_batch
        self.count_tokens = count_tokens
        self.retry_on = retry_on
        self.max_retries = max_retries
        self._sleep = sleep
        self.on_retry = on_retry

    def _request(self, tokens: int, call: Callable[[], T]) -> T:
        attempt = 0
        while True:
            self.limiter.acquire(tokens)
            try:
                return call()
            except self.retry_on:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                if self.on_retry:
                    self.on_retry(attempt)
                self._sleep(WINDOW_SECONDS)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each distinct text once; duplicates in the same call share a vector."""
        unique = list(dict.fromkeys(texts))
        vectors: dict[str, list[float]] = {}
        for batch, tokens in batch_by_tokens(unique, self.batch_tokens, self.count_tokens):
            embedded = self._request(tokens, lambda b=batch: self.inner.embed_documents(b))
            vectors.update(zip(batch, embedded, strict=True))
            if self.on_batch:
                self.on_batch(len(batch))
        return [vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._request(self.count_tokens(text), lambda: self.inner.embed_query(text))


def voyage_token_counter(model: str) -> TokenCounter:
    """Exact Voyage token counts, computed locally with the model's tokenizer."""
    client = voyageai.Client()  # reads VOYAGE_API_KEY
    return lambda text: client.count_tokens([text], model=model)


def cache_namespace(settings: Settings) -> str:
    """Cache keys include the provider, model and dimensions, so changing any re-embeds."""
    return f"{slug(settings.embed_model)}-{settings.embed_dimensions}"


def build_ingest_embeddings(
    settings: Settings,
    inner: Embeddings | None = None,
    on_batch: Callable[[int], None] | None = None,
    on_retry: Callable[[int], None] | None = None,
) -> CacheBackedEmbeddings:
    """Cache(RateLimited(EMBED_MODEL)): only cache misses reach the API, within rate limits.

    Voyage batches are sized with its real tokenizer and retried on its rate-limit
    errors. Other providers, and a custom `inner` (tests), use the character estimate.
    """
    provider, name = parse_spec(settings.embed_model)
    count: TokenCounter = estimate_tokens
    retry_on: tuple[type[Exception], ...] = ()
    if inner is None and provider == VOYAGE:
        count, retry_on = voyage_token_counter(name), (RateLimitError,)
    if inner is None:
        inner = embeddings(settings.embed_model, settings.embed_dimensions)
    limited = RateLimitedEmbeddings(
        inner,
        RateLimiter(settings.embed_rpm, settings.embed_tpm),
        settings.embed_batch_tokens,
        on_batch,
        count_tokens=count,
        retry_on=retry_on,
        on_retry=on_retry,
    )
    store = LocalFileStore(Path(settings.cache_dir) / "embeddings")
    return CacheBackedEmbeddings.from_bytes_store(
        limited, store, namespace=cache_namespace(settings), key_encoder="sha256"
    )
