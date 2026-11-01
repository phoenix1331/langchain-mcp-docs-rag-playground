from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import Field

from app.config import Settings
from ingest import embeddings as embeddings_module
from ingest.embeddings import (
    RateLimitedEmbeddings,
    RateLimiter,
    batch_by_tokens,
    build_ingest_embeddings,
    cache_namespace,
    estimate_tokens,
    voyage_token_counter,
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
        "router_model": "fake:r",
        "answer_model": "fake:a",
        "embed_model": "voyageai:voyage-test",
        "rerank_model": "fake:rr",
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

    assert batches == [(["a" * 40, "b" * 40], 20), (["c" * 40], 10), (["d" * 400], 100)]
    assert batch_by_tokens([], max_tokens=20) == []


def test_batch_by_tokens_uses_the_given_counter() -> None:
    batches = batch_by_tokens(["a", "b", "c"], max_tokens=10, count=lambda text: 6)

    assert batches == [(["a"], 6), (["b"], 6), (["c"], 6)]


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


def test_cache_namespace_includes_model_and_dimensions(tmp_path: Path) -> None:
    assert cache_namespace(settings(tmp_path, embed_dimensions=512)) == "voyageai-voyage-test-512"


def test_cache_only_sends_misses_to_the_api(tmp_path: Path) -> None:
    inner = CountingEmbeddings(size=8)
    config = settings(tmp_path)

    first = build_ingest_embeddings(config, inner).embed_documents(["same", "other"])
    second = build_ingest_embeddings(config, inner).embed_documents(["same", "new"])

    assert inner.calls == [["same", "other"], ["new"]]
    assert first[0] == second[0]
    assert (tmp_path / "embeddings").is_dir()


class Flaky(DeterministicFakeEmbedding):
    failures: int = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.failures:
            self.failures -= 1
            raise TimeoutError("rate limited")
        return super().embed_documents(texts)


def test_rate_limit_errors_are_retried_after_a_window(clock: FakeClock) -> None:
    retries: list[int] = []
    embeddings = RateLimitedEmbeddings(
        Flaky(size=8, failures=2),
        RateLimiter(0, 0, clock=clock),
        1000,
        retry_on=(TimeoutError,),
        sleep=clock.sleep,
        on_retry=retries.append,
    )

    assert len(embeddings.embed_documents(["a"])) == 1
    assert retries == [1, 2]
    assert clock.sleeps == [60, 60]


def test_retries_give_up_after_the_limit(clock: FakeClock) -> None:
    embeddings = RateLimitedEmbeddings(
        Flaky(size=8, failures=5),
        RateLimiter(0, 0, clock=clock),
        1000,
        retry_on=(TimeoutError,),
        max_retries=2,
        sleep=clock.sleep,
    )

    with pytest.raises(TimeoutError):
        embeddings.embed_documents(["a"])
    assert clock.sleeps == [60, 60]


def test_other_errors_are_not_retried(clock: FakeClock) -> None:
    embeddings = RateLimitedEmbeddings(
        Flaky(size=8, failures=1), RateLimiter(0, 0, clock=clock), 1000, sleep=clock.sleep
    )

    with pytest.raises(TimeoutError):
        embeddings.embed_documents(["a"])
    assert clock.sleeps == []


def test_voyage_token_counter_uses_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], str]] = []

    class FakeClient:
        def count_tokens(self, texts: list[str], model: str) -> int:
            calls.append((texts, model))
            return 42

    monkeypatch.setattr(embeddings_module.voyageai, "Client", FakeClient)

    assert voyage_token_counter("voyage-test")("hello") == 42
    assert calls == [(["hello"], "voyage-test")]


def test_build_uses_voyage_counter_and_retries_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = DeterministicFakeEmbedding(size=8)
    monkeypatch.setattr(embeddings_module, "embeddings", lambda spec, dims: fake)
    monkeypatch.setattr(embeddings_module, "voyage_token_counter", lambda name: lambda text: 7)

    built = build_ingest_embeddings(settings(tmp_path))

    limited = built.underlying_embeddings
    assert isinstance(limited, RateLimitedEmbeddings)
    assert limited.inner is fake
    assert limited.count_tokens("anything") == 7
    assert limited.retry_on == (embeddings_module.RateLimitError,)


def test_other_providers_use_the_estimate_and_no_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = DeterministicFakeEmbedding(size=8)
    seen: list[tuple[str, int]] = []

    def build(spec: str, dims: int) -> DeterministicFakeEmbedding:
        seen.append((spec, dims))
        return fake

    monkeypatch.setattr(embeddings_module, "embeddings", build)

    built = build_ingest_embeddings(settings(tmp_path, embed_model="openai:text-embedding-3-small"))

    limited = built.underlying_embeddings
    assert isinstance(limited, RateLimitedEmbeddings)
    assert seen == [("openai:text-embedding-3-small", 1024)]
    assert limited.count_tokens is estimate_tokens
    assert limited.retry_on == ()
