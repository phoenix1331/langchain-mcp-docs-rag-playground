from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app import main
from app.main import MAX_QUESTION_CHARS, Services, create_app, render_markdown, sse
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
    """Streams like a compiled graph: task start/result events, answer tokens, final values."""

    def __init__(self, result: dict[str, Any] | Exception = RESULT) -> None:
        self.result = result
        self.calls: list[dict[str, str]] = []

    async def astream(
        self, inputs: dict[str, str], stream_mode: list[str]
    ) -> AsyncIterator[tuple[str, Any]]:
        assert stream_mode == ["tasks", "messages", "values"]
        self.calls.append(inputs)
        state = {**inputs, "query": inputs["question"]}
        yield "tasks", {"id": "1", "name": "route", "input": state}
        yield "messages", (AIMessageChunk(content="ignored"), {"langgraph_node": "route"})
        yield "tasks", {"id": "1", "name": "route", "result": {"trace": [RESULT["trace"][0]]}}
        if isinstance(self.result, Exception):
            raise self.result
        yield "tasks", {"id": "2", "name": "answer", "input": {**state, "route": "direct"}}
        for text in ["Use the ", "<b>grant</b>", "\n", ""]:
            yield "messages", (AIMessageChunk(content=text), {"langgraph_node": "answer"})
        yield "tasks", {"id": "2", "name": "answer", "result": None}
        yield "values", self.result


def parse_events(body: str) -> list[tuple[str, str]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        name = lines[0].removeprefix("event: ")
        data = "\n".join(line.removeprefix("data: ") for line in lines[1:])
        events.append((name, data))
    return events


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


def test_ask_returns_the_stream_shell_without_running_the_graph(
    client: TestClient, graph: FakeGraph
) -> None:
    response = client.post(
        "/ask", data={"question": "  How are tokens refreshed?  ", "version": "2026-07-28"}
    )

    assert response.status_code == 200
    assert graph.calls == []
    assert (
        'sse-connect="/ask/stream?question=How+are+tokens+refreshed%3F&amp;version=2026-07-28"'
        in response.text
    )
    assert 'sse-close="done"' in response.text
    assert '<p class="asked">How are tokens refreshed?</p>' in response.text
    assert "<html" not in response.text  # a fragment, not a page


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
def test_ask_validates(client: TestClient, data: dict[str, str], message: str) -> None:
    response = client.post("/ask", data=data)

    assert response.status_code == 422
    assert message in response.text


def test_stream_sends_stages_tokens_then_the_rendered_answer(
    client: TestClient, graph: FakeGraph
) -> None:
    response = client.get("/ask/stream", params={"question": " q ", "version": "all"})

    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert graph.calls == [{"question": "q", "spec_version": "all"}]
    events = parse_events(response.text)
    assert [name for name, _ in events] == [
        "stage", "stage", "stage", "token", "token", "token", "stage", "done"
    ]  # fmt: skip
    assert events[0][1] == '<li class="stage">Deciding whether this needs the docs...</li>'
    assert events[1][1] == '<li class="stage-done">done in 0.9s, 200 tokens</li>'
    assert events[2][1] == '<li class="stage">Replying...</li>'
    assert [data for name, data in events if name == "token"] == [
        "Use the ",
        "&lt;b&gt;grant&lt;/b&gt;",
        "\n",
    ]
    assert events[6][1] == '<li class="stage-done">done</li>'
    done = events[-1][1]
    assert "<strong>refresh grant</strong> [1]" in done
    assert "<script>" not in done  # model output cannot inject markup
    assert "2 steps, 2.5s, 1200 tokens" in done


def test_stream_reports_validation_errors_as_done(client: TestClient, graph: FakeGraph) -> None:
    response = client.get("/ask/stream", params={"question": "q", "version": "1999-01-01"})

    assert response.status_code == 200  # a non-200 would make EventSource retry
    assert parse_events(response.text) == [
        ("done", '<p class="error" role="alert">Unknown spec version &#39;1999-01-01&#39;.</p>')
    ]
    assert graph.calls == []


def test_stream_reports_provider_failures_as_done(client: TestClient, graph: FakeGraph) -> None:
    graph.result = RuntimeError("429 RESOURCE_EXHAUSTED")

    response = client.get("/ask/stream", params={"question": "q", "version": "all"})

    events = parse_events(response.text)

    assert events[-1][0] == "done"
    assert "may be rate limiting" in events[-1][1]


def test_sse_splits_multiline_data() -> None:
    assert sse("done", "a\nb") == "event: done\ndata: a\ndata: b\n\n"


def test_page_loads_the_sse_extension_and_retry_guard(client: TestClient) -> None:
    page = client.get("/").text

    assert "htmx-ext-sse@2.2.4" in page
    assert "/static/app.js" in page
    assert "htmx:sseError" in client.get("/static/app.js").text


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
