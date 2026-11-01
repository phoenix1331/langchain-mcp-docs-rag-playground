from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from app.config import MissingSecretError, Settings
from app.retrieval import cli
from app.retrieval.bm25 import KeywordIndex, MissingSnapshotError
from app.retrieval.search import Search
from tests.retrieval_fakes import ReverseReranker, StaticRetriever, doc

CHUNKS = [
    doc("new", "2026-07-28", "Lifecycle › Initialization"),
    doc("guide", "unversioned", "Guides › Initialization"),
]


@pytest.fixture
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://u:p@localhost/db",
        router_model="r",
        answer_model="a",
        embed_model="e",
        rerank_model="rr",
        voyage_api_key=SecretStr("key"),
        cache_dir=tmp_path,
    )
    engine = MagicMock()
    store = MagicMock()
    store.as_retriever.return_value = StaticRetriever(docs=[CHUNKS[0]])
    search = Search(keyword=KeywordIndex(CHUNKS), store=store, reranker=ReverseReranker(), top_k=5)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "make_engine", lambda url: engine)
    monkeypatch.setattr(cli.Search, "from_settings", lambda s, e: search)
    return engine


def test_query_prints_ranked_hits(wired: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["initialization"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("hybrid + rerank, latest -> 2026-07-28 + unversioned: 2 results in ")
    assert " 1. 1.000  " in out
    assert "https://example.test/new" in out
    assert "bm25 #" in out and "vector #1" in out
    assert "body of new" in out
    wired.dispose.assert_called_once()


def test_query_without_rerank_shows_no_score(
    wired: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["initialization", "--mode", "bm25", "--no-rerank", "--version", "all"]

    assert cli.main(args) == 0

    out = capsys.readouterr().out
    assert out.startswith("bm25, all: ")
    assert " 1.   -    " in out
    assert "#" not in out


@pytest.mark.parametrize(
    "error",
    [
        MissingSnapshotError("run `make ingest` first"),
        MissingSecretError("VOYAGE_API_KEY is not set"),
    ],
)
def test_query_reports_setup_errors(
    wired: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def fail(settings: Settings, engine: object) -> Search:
        raise error

    monkeypatch.setattr(cli.Search, "from_settings", fail)

    assert cli.main(["q"]) == 1
    assert capsys.readouterr().err == f"error: {error}\n"
    wired.dispose.assert_called_once()


def test_query_rejects_unknown_version(
    wired: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["q", "--version", "1999-01-01"]) == 1
    assert "unknown spec version" in capsys.readouterr().err


def test_snippet_strips_breadcrumb_and_truncates() -> None:
    long = doc("x", heading="A › B")
    long.page_content = "A › B\n\n" + "word " * 100

    text = cli.snippet(long)

    assert not text.startswith("A ›")
    assert len(text) == cli.SNIPPET_CHARS and text.endswith("...")
    assert cli.snippet(doc("y")) == "body of y"
