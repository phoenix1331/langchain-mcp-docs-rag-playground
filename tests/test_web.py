from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import MAX_QUESTION_CHARS, Services, create_app, render_markdown
from ingest.fetch import SpecVersions

VERSIONS = SpecVersions(("2025-11-25", "2026-07-28"))
RESULT = {
    "answer": "Use the **refresh grant** [1].\n\n<script>alert(1)</script>",
    "trace": [
        {"step": "route", "summary": "", "ms": 900, "tokens": 200},
        {"step": "answer", "summary": "", "ms": 1600, "tokens": 1000},
    ],
}


class FakeGraph:
    def __init__(self, result: dict[str, Any] | Exception = RESULT) -> None:
        self.result = result
        self.calls: list[dict[str, str]] = []

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        self.calls.append(inputs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture
def closed() -> list[bool]:
    return []


@pytest.fixture
def client(graph: FakeGraph, closed: list[bool]) -> Iterator[TestClient]:
    services = Services(versions=VERSIONS, graph=graph, close=lambda: closed.append(True))
    with TestClient(create_app(lambda: services)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_index_lists_versions_newest_first_with_latest_selected(client: TestClient) -> None:
    page = client.get("/").text

    assert page.index('value="2026-07-28" selected') < page.index('value="2025-11-25"')
    assert "2026-07-28 (latest)" in page
    assert '<option value="all">All versions</option>' in page
    assert 'hx-post="/ask"' in page
    assert f'maxlength="{MAX_QUESTION_CHARS}"' in page


def test_static_css_is_served(client: TestClient) -> None:
    assert client.get("/static/app.css").status_code == 200


def test_ask_renders_the_answer_fragment(client: TestClient, graph: FakeGraph) -> None:
    response = client.post(
        "/ask", data={"question": "  How are tokens refreshed?  ", "version": "2026-07-28"}
    )

    assert response.status_code == 200
    assert graph.calls == [{"question": "How are tokens refreshed?", "spec_version": "2026-07-28"}]
    assert "<strong>refresh grant</strong> [1]" in response.text
    assert "<script>" not in response.text  # model output cannot inject markup
    assert "2 steps, 2.5s, 1200 tokens" in response.text
    assert "<html" not in response.text  # a fragment, not a page


def test_ask_accepts_all_versions(client: TestClient, graph: FakeGraph) -> None:
    client.post("/ask", data={"question": "q", "version": "all"})

    assert graph.calls[0]["spec_version"] == "all"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"question": "   ", "version": "all"}, "Ask a question of 1 to 1000 characters."),
        (
            {"question": "x" * (MAX_QUESTION_CHARS + 1), "version": "all"},
            "Ask a question of 1 to 1000 characters.",
        ),
        ({"question": "q", "version": "1999-01-01"}, "Unknown spec version &#39;1999-01-01&#39;."),
    ],
)
def test_ask_validates_without_calling_the_graph(
    client: TestClient, graph: FakeGraph, data: dict[str, str], message: str
) -> None:
    response = client.post("/ask", data=data)

    assert response.status_code == 422
    assert message in response.text
    assert graph.calls == []


def test_ask_reports_provider_failures(client: TestClient, graph: FakeGraph) -> None:
    graph.result = RuntimeError("429 RESOURCE_EXHAUSTED")

    response = client.post("/ask", data={"question": "q", "version": "all"})

    assert response.status_code == 502
    assert "may be rate limiting" in response.text


def test_shutdown_closes_services(graph: FakeGraph) -> None:
    closed: list[bool] = []
    services = Services(versions=VERSIONS, graph=graph, close=lambda: closed.append(True))

    with TestClient(create_app(lambda: services)):
        pass

    assert closed == [True]


def test_render_markdown_escapes_html_and_keeps_tables() -> None:
    html = render_markdown("| a |\n| - |\n| <b>x</b> |")

    assert "<table>" in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html


def test_build_services_wires_search_and_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = MagicMock()
    search = MagicMock()
    search.keyword.versions = VERSIONS
    monkeypatch.setattr(main, "get_settings", lambda: MagicMock(database_url="db"))
    monkeypatch.setattr(main, "make_engine", lambda url: engine)
    monkeypatch.setattr(main.Search, "from_settings", lambda s, e: search)
    monkeypatch.setattr(main, "build_question_graph", lambda s, sr: "graph")

    services = main.build_services()
    services.close()

    assert (services.versions, services.graph) == (VERSIONS, "graph")
    engine.dispose.assert_called_once()


def test_default_services_close_is_a_no_op() -> None:
    Services(versions=VERSIONS, graph=FakeGraph()).close()
