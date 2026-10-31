from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import Field, SecretStr

from app.config import MissingSecretError, Settings
from ingest.embeddings import (
    RateLimitedEmbeddings,
    RateLimiter,
    batch_by_tokens,
    build_ingest_embeddings,
    cache_namespace,
    estimate_tokens,
    voyage_embeddings,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class CountingEmbeddings(DeterministicFakeEmbedding):
    calls: list[list[str]] = Field(default_factory=list)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return super().embed_documents(texts)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql+psycopg://u:p@localhost/db",
        "router_model": "r",
        "answer_model": "a",
        "embed_model": "voyage-test",
        "rerank_model": "rr",
        "voyage_api_key": SecretStr("key"),
        "cache_dir": tmp_path,
    }
    return Settings(_env_file=None, **{**values, **overrides})


def test_estimate_tokens_is_at_least_one() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 40) == 10


class TestRateLimiter:
    def test_unlimited_never_sleeps(self, clock: FakeClock) -> None:
        limiter = RateLimiter(0, 0, clock=clock, sleep=clock.sleep)

        for _ in range(50):
            limiter.acquire(10_000)

        assert clock.sleeps == []

    def test_requests_per_minute(self, clock: FakeClock) -> None:
        limiter = RateLimiter(2, 0, clock=clock, sleep=clock.sleep)

        limiter.acquire(1)
        clock.now = 5
        limiter.acquire(1)
        limiter.acquire(1)

        assert clock.sleeps == [55]
        assert clock.now == 60

    def test_tokens_per_minute(self, clock: FakeClock) -> None:
        limiter = RateLimiter(0, 100, clock=clock, sleep=clock.sleep)

        limiter.acquire(70)
        limiter.acquire(30)
        limiter.acquire(1)

        assert clock.sleeps == [60]

    def test_oversized_request_is_allowed_when_window_is_empty(self, clock: FakeClock) -> None:
        limiter = RateLimiter(0, 100, clock=clock, sleep=clock.sleep)

        limiter.acquire(500)

        assert clock.sleeps == []

    def test_sleep_has_a_floor(self, clock: FakeClock) -> None:
        limiter = RateLimiter(1, 0, clock=clock, sleep=clock.sleep)
        limiter.acquire(1)
        clock.now = 59.999

        limiter.acquire(1)

        assert clock.sleeps[0] == pytest.approx(0.01)


def test_batch_by_tokens() -> None:
    texts = ["a" * 40, "b" * 40, "c" * 40, "d" * 400]

    batches = batch_by_tokens(texts, max_tokens=20)

    assert batches == [["a" * 40, "b" * 40], ["c" * 40], ["d" * 400]]
    assert batch_by_tokens([], max_tokens=20) == []


def test_rate_limited_embeddings_batches_and_reports(clock: FakeClock) -> None:
    inner = CountingEmbeddings(size=8)
    reported: list[int] = []
    embeddings = RateLimitedEmbeddings(
        inner, RateLimiter(1, 0, clock=clock, sleep=clock.sleep), 20, reported.append
    )

    vectors = embeddings.embed_documents(["a" * 40, "b" * 40, "c" * 40])

    assert len(vectors) == 3
    assert inner.calls == [["a" * 40, "b" * 40], ["c" * 40]]
    assert reported == [2, 1]
    assert clock.sleeps == [60]


def test_duplicate_texts_are_embedded_once(clock: FakeClock) -> None:
    inner = CountingEmbeddings(size=8)
    embeddings = RateLimitedEmbeddings(inner, RateLimiter(0, 0, clock=clock), 1000)

    vectors = embeddings.embed_documents(["same", "other", "same"])

    assert inner.calls == [["same", "other"]]
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]


def test_rate_limited_embed_query_uses_the_limiter(clock: FakeClock) -> None:
    limiter = RateLimiter(1, 0, clock=clock, sleep=clock.sleep)
    embeddings = RateLimitedEmbeddings(DeterministicFakeEmbedding(size=8), limiter, 20)

    embeddings.embed_query("one")
    embeddings.embed_query("two")

    assert len(embeddings.embed_query("three")) == 8
    assert clock.sleeps == [60, 60]


def test_voyage_embeddings_uses_settings(tmp_path: Path) -> None:
    embeddings = voyage_embeddings(settings(tmp_path))

    assert embeddings.model == "voyage-test"
    assert embeddings.output_dimension == 1024


def test_voyage_embeddings_requires_key(tmp_path: Path) -> None:
    with pytest.raises(MissingSecretError, match="VOYAGE_API_KEY"):
        voyage_embeddings(settings(tmp_path, voyage_api_key=None))


def test_cache_namespace_includes_model_and_dimensions(tmp_path: Path) -> None:
    assert cache_namespace(settings(tmp_path, embed_dimensions=512)) == "voyage-test-512"


def test_cache_only_sends_misses_to_the_api(tmp_path: Path) -> None:
    inner = CountingEmbeddings(size=8)
    config = settings(tmp_path)

    first = build_ingest_embeddings(config, inner).embed_documents(["same", "other"])
    second = build_ingest_embeddings(config, inner).embed_documents(["same", "new"])

    assert inner.calls == [["same", "other"], ["new"]]
    assert first[0] == second[0]
    assert (tmp_path / "embeddings").is_dir()
