from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.indexing import InMemoryRecordManager
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import SecretStr

from app.config import MissingSecretError, Settings
from ingest import cli
from ingest.cli import IngestSummary, format_summary, run_ingest
from ingest.fetch import FetchError
from ingest.index import DimensionMismatchError

PAGE = "---\ntitle: Lifecycle\n---\n## Init\n\nThe client sends initialize first.\n"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql+psycopg://u:p@localhost/db",
        router_model="r",
        answer_model="a",
        embed_model="fake",
        rerank_model="rr",
        voyage_api_key=SecretStr("key"),
        cache_dir=tmp_path,
    )


@pytest.fixture
def fake_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    repo = tmp_path / cli.REPO_DIR_NAME
    for rel in [
        "docs/specification/2025-11-25/basic/lifecycle.mdx",
        "docs/specification/2026-07-28/basic/lifecycle.mdx",
    ]:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(PAGE)
    (repo / "docs/docs.json").write_text("{}")

    manager = InMemoryRecordManager("test")
    manager.create_schema()
    engine = MagicMock()
    monkeypatch.setattr(cli, "sync_repo", lambda url, dest: "abc123def4567890")
    monkeypatch.setattr(cli, "make_engine", lambda url: engine)
    monkeypatch.setattr(cli, "build_record_manager", lambda e: manager)
    monkeypatch.setattr(cli, "check_dimensions", lambda e, dims: None)
    monkeypatch.setattr(cli, "build_vector_store", lambda e, emb, dims: InMemoryVectorStore(emb))
    return make_settings(tmp_path)


def test_run_ingest_indexes_then_skips(fake_pipeline: Settings) -> None:
    logs: list[str] = []
    fake: Embeddings = DeterministicFakeEmbedding(size=8)

    first = run_ingest(fake_pipeline, fake, logs.append)
    second = run_ingest(fake_pipeline, fake, logs.append)

    assert (first.pages, first.chunks, first.added) == (2, 2, 2)
    # Identical text in two versions: the second copy comes from the embedding cache.
    assert first.embedded == 1
    assert (second.added, second.skipped, second.embedded) == (0, 2, 0)
    assert first.versions == ("2025-11-25", "2026-07-28")
    assert any("2 pages -> 2 chunks" in line for line in logs)
    assert any("embedded 1 new texts" in line for line in logs)


def test_format_summary() -> None:
    summary = IngestSummary(
        commit="abc123def4567890",
        versions=("2025-11-25", "2026-07-28"),
        pages=2,
        chunks=10,
        added=3,
        updated=0,
        skipped=7,
        deleted=1,
        embedded=2,
        seconds=1.234,
    )

    text = format_summary(summary)

    assert "commit:   abc123def456 (latest spec 2026-07-28)" in text
    assert "skipped:  7" in text
    assert "took:     1.2s" in text


def test_main_prints_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    summary = IngestSummary("abc123def4567890", ("2026-07-28",), 1, 1, 1, 0, 0, 0, 1, 0.1)
    monkeypatch.setattr(cli, "get_settings", lambda: make_settings(tmp_path))
    monkeypatch.setattr(cli, "run_ingest", lambda settings: summary)

    assert cli.main() == 0
    assert "added:    1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error",
    [
        FetchError("git clone failed: nope"),
        MissingSecretError("VOYAGE_API_KEY is not set"),
        DimensionMismatchError("vector(8) but EMBED_DIMENSIONS is 1024"),
    ],
)
def test_main_reports_known_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error: Exception,
) -> None:
    def fail(settings: Settings) -> IngestSummary:
        raise error

    monkeypatch.setattr(cli, "get_settings", lambda: make_settings(tmp_path))
    monkeypatch.setattr(cli, "run_ingest", fail)

    assert cli.main() == 1
    assert f"error: {error}" in capsys.readouterr().err


def test_default_log_writes_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    cli._log("indexed 100/200 chunks")

    assert capsys.readouterr().err == "indexed 100/200 chunks\n"


def test_rate_limit_retries_are_logged(
    fake_pipeline: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_build = cli.build_ingest_embeddings

    def build(
        settings: Settings,
        inner: Embeddings | None,
        on_batch: Callable[[int], None],
        on_retry: Callable[[int], None],
    ) -> CacheBackedEmbeddings:
        on_retry(1)
        return real_build(settings, inner, on_batch, on_retry)

    monkeypatch.setattr(cli, "build_ingest_embeddings", build)
    logs: list[str] = []

    run_ingest(fake_pipeline, DeterministicFakeEmbedding(size=8), logs.append)

    assert "  rate limited by the API; waiting a minute (retry 1)" in logs
