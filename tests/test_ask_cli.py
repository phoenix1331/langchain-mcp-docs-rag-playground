from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.graph import cli
from app.models import ModelConfigError
from app.retrieval.bm25 import MissingSnapshotError

RESULT: dict[str, Any] = {
    "answer": "Refresh with the grant [1].\n",
    "citations": [
        {
            "number": 1,
            "chunk_id": "a",
            "heading_path": "Auth › Refresh",
            "spec_version": "2026-07-28",
            "source_url": "https://example.test/a",
        }
    ],
    "trace": [
        {"step": "route", "summary": "retrieve: about MCP", "ms": 900, "tokens": 200},
        {"step": "retrieve", "summary": "6 chunks", "ms": 1400, "tokens": 0},
        {"step": "answer", "summary": "cited 1 of 6 sources", "ms": 3200, "tokens": 2100},
    ],
}


@pytest.fixture
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://u:p@localhost/db",
        router_model="fake:r",
        answer_model="fake:a",
        embed_model="fake:e",
        rerank_model="fake:rr",
        cache_dir=tmp_path,
    )
    engine = MagicMock()
    search = MagicMock()
    graph = MagicMock()
    graph.invoke.return_value = RESULT
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "make_engine", lambda url: engine)
    monkeypatch.setattr(cli.Search, "from_settings", lambda s, e: search)
    monkeypatch.setattr(cli, "build_question_graph", lambda s, sr: graph)
    return {"engine": engine, "search": search, "graph": graph}


def test_ask_prints_answer_sources_and_trace(
    wired: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["How are tokens refreshed?", "--version", "2026-07-28"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("Refresh with the grant [1].\n\nSources\n")
    assert "  [1] 2026-07-28  Auth › Refresh\n      https://example.test/a" in out
    assert "  route       0.9s    200 tok  retrieve: about MCP" in out
    assert "  total 5.5s, 2300 tokens" in out
    wired["graph"].invoke.assert_called_once_with(
        {"question": "How are tokens refreshed?", "spec_version": "2026-07-28"}
    )
    wired["search"].scope.assert_called_once_with("2026-07-28")
    wired["engine"].dispose.assert_called_once()


def test_ask_without_citations_skips_sources(
    wired: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    wired["graph"].invoke.return_value = {**RESULT, "citations": []}

    assert cli.main(["hi"]) == 0
    assert "Sources" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "error",
    [
        MissingSnapshotError("run `make ingest` first"),
        ModelConfigError("no reranker for provider 'x'"),
        ValueError("unknown spec version '1999-01-01'"),
    ],
)
def test_ask_reports_setup_errors(
    wired: dict[str, Any], capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    wired["search"].scope.side_effect = error

    assert cli.main(["q", "--version", "1999-01-01"]) == 1
    assert capsys.readouterr().err == f"error: {error}\n"
    wired["graph"].invoke.assert_not_called()
    wired["engine"].dispose.assert_called_once()
