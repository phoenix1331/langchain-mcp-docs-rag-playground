import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app import main
from app.main import (
    DEFAULT_REPO_URL,
    MAX_QUESTION_CHARS,
    Services,
    create_app,
    link_citations,
    render_markdown,
    sse,
)
from ingest.fetch import SpecVersions

VERSIONS = SpecVersions(("2025-11-25", "2026-07-28"))
CITATION = {
    "number": 1,
    "chunk_id": "a",
    "heading_path": "Base Protocol › Authorization › Refresh Tokens",
    "spec_version": "2026-07-28",
    "source_url": "https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization",
    "source_path": "docs/specification/2026-07-28/basic/authorization/index.mdx",
}
RESULT = {
    "answer": "Use the **refresh grant** [1].\n\n<script>alert(1)</script>",
    "citations": [CITATION],
    "trace": [
        {"step": "route", "summary": "", "ms": 900, "tokens": 200},
        {
            "step": "grade",
            "summary": "1 of 2 chunks relevant",
            "ms": 0,
            "tokens": 0,
            "details": ["relevant: Auth › Refresh", "not relevant: <b>Other</b>"],
        },
        {"step": "answer", "summary": "cited 1", "ms": 1600, "tokens": 1000},
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
    # The question box is cleared after a successful submit, but kept after a 422.
    assert "if (event.detail.successful) this.querySelector('textarea').value = ''" in page


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
    assert '<strong>refresh grant</strong> [<a class="cite" href="#source-1">1</a>]' in done
    assert '<h2 id="sources-heading">Sources</h2>' in done
    assert '<li id="source-1">' in done
    assert "Base Protocol › Authorization › Refresh Tokens" in done
    assert f'href="{CITATION["source_url"]}"' in done
    assert f'href="{DEFAULT_REPO_URL}/blob/main/{CITATION["source_path"]}"' in done
    assert "<script>" not in done  # model output cannot inject markup
    assert '<details class="trace">' in done
    assert "3 steps, 2.5s, 1200 tokens" in done
    assert '<span class="trace-name">grade</span>' in done
    assert "<li>relevant: Auth › Refresh</li>" in done
    assert "&lt;b&gt;Other&lt;/b&gt;" in done  # details are escaped
    assert "0.0s</span>" in done  # no token count when a step used none
    assert '<p class="trace-summary">cited 1</p>' in done


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
    settings = MagicMock(database_url="db", mcp_repo_url="https://github.com/org/repo.git")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "make_engine", lambda url: engine)
    monkeypatch.setattr(main.Search, "from_settings", lambda s, e: search)
    monkeypatch.setattr(main, "build_question_graph", lambda s, sr: "graph")

    services = main.build_services()
    services.close()

    assert (services.versions, services.graph) == (VERSIONS, "graph")
    assert services.repo_url == "https://github.com/org/repo"
    engine.dispose.assert_called_once()


def test_default_services_close_is_a_no_op() -> None:
    Services(versions=VERSIONS, graph=FakeGraph()).close()


def test_declines_title_the_panel_closest_sections(client: TestClient, graph: FakeGraph) -> None:
    graph.result = {
        "answer": "The MCP docs for 2026-07-28 don't cover this.\n\n- [1] Closest",
        "citations": [CITATION],
        "trace": [{"step": "decline", "summary": "", "ms": 0, "tokens": 0}],
    }

    response = client.get("/ask/stream", params={"question": "q", "version": "all"})

    done = parse_events(response.text)[-1][1]
    assert '<h2 id="sources-heading">Closest sections</h2>' in done


def test_answers_without_citations_have_no_panel(client: TestClient, graph: FakeGraph) -> None:
    graph.result = {**RESULT, "citations": []}

    response = client.get("/ask/stream", params={"question": "q", "version": "all"})

    assert "sources-heading" not in parse_events(response.text)[-1][1]


def cite(n: int) -> str:
    return f'<a class="cite" href="#source-{n}">{n}</a>'


CODE_SPANS = "<p><code>items[0]</code> and <code>a[1]</code></p>"
CODE_BLOCK = "<pre><code>x = y[1]\n</code></pre>"


@pytest.mark.parametrize(
    ("rendered", "numbers", "expected"),
    [
        ("<p>A [1].</p>", [1], f"<p>A [{cite(1)}].</p>"),
        ("<p>B [1][3]</p>", [1, 3], f"<p>B [{cite(1)}][{cite(3)}]</p>"),
        ("<p>C [3, 1]</p>", [1, 3], f"<p>C [{cite(3)}, {cite(1)}]</p>"),
        ("<p>Not cited [7]</p>", [1], "<p>Not cited [7]</p>"),
        ("<p>Partly [1, 7]</p>", [1], "<p>Partly [1, 7]</p>"),
        (CODE_SPANS, [1], CODE_SPANS),
        (f"{CODE_BLOCK}<p>[1]</p>", [1], f"{CODE_BLOCK}<p>[{cite(1)}]</p>"),
    ],
)
def test_link_citations(rendered: str, numbers: list[int], expected: str) -> None:
    assert link_citations(rendered, numbers) == expected


def eval_run(label: str, complete: bool = True) -> dict[str, Any]:
    return {
        "label": label,
        "started_at": "20261101T203045Z",
        "complete": complete,
        "config": {"mode": "hybrid", "rerank": True, "grading": True, "answer_model": "fake:a"},
        "summary": {
            "questions": 2,
            "pass_rate": 0.5,
            "retrieval_hit_rate": 1.0,
            "correctness": 0.75,
            "citation_accuracy": 1.0,
            "correct_declines": None,
            "false_declines": 0,
            "mean_seconds": 4.2,
            "total_tokens": 1234,
        },
        "results": [
            {
                "id": "good-one",
                "category": "version_specific",
                "spec_version": "2026-07-28",
                "outcome": "answered",
                "passed": True,
                "error": None,
                "question": "What is it?",
                "answer": "It is **this** [1]. <script>x</script>",
                "facts_found": [True, True],
                "retrieval_hit": True,
                "judge_reason": "both facts stated",
                "retrieved": ["docs/a.mdx"],
                "cited": ["docs/a.mdx"],
                "seconds": 4.0,
            },
            {
                "id": "broken-one",
                "category": "cross_version",
                "spec_version": "all",
                "passed": False,
                "error": "GoogleAPIError: 503 UNAVAILABLE",
                "question": "What changed?",
            },
        ],
    }


@pytest.fixture
def eval_client(tmp_path: Path, graph: FakeGraph) -> Iterator[TestClient]:
    (tmp_path / "20261101T203045Z-baseline.json").write_text(json.dumps(eval_run("baseline")))
    (tmp_path / "20261102T090000Z-partial.json").write_text(json.dumps(eval_run("partial", False)))
    (tmp_path / ".gitkeep").write_text("")
    services = Services(versions=VERSIONS, graph=graph, results_dir=tmp_path)
    with TestClient(create_app(lambda: services)) as test_client:
        yield test_client


def test_evals_lists_runs_newest_first(eval_client: TestClient) -> None:
    page = eval_client.get("/evals").text

    assert page.index("partial</a>") < page.index("baseline</a>")
    assert '<a href="/evals/20261101T203045Z-baseline">baseline</a>' in page
    assert '<span class="badge">partial</span>' in page
    assert "2026-11-01 20:30 UTC" in page
    assert "hybrid + rerank + grading" in page
    assert "fake:a" in page
    assert '<td class="num">50%</td>' in page
    assert '<td class="num">-</td>' in page  # correct_declines is None
    assert '<td class="num">4.2s</td>' in page
    assert '<a href="/evals">Evals</a>' in page


def test_evals_page_without_runs(tmp_path: Path, graph: FakeGraph) -> None:
    services = Services(versions=VERSIONS, graph=graph, results_dir=tmp_path)
    with TestClient(create_app(lambda: services)) as test_client:
        assert "No eval runs yet." in test_client.get("/evals").text


def test_eval_run_shows_every_question(eval_client: TestClient) -> None:
    page = eval_client.get("/evals/20261101T203045Z-baseline").text

    assert "<dt>answer_model</dt><dd>fake:a</dd>" in page
    assert '<span class="result-id">good-one</span>' in page
    assert '<span class="result-id">broken-one</span>' in page
    assert "facts 2/2" in page
    assert "retrieved" in page
    assert "<strong>this</strong> [1]" in page
    assert "<script>x" not in page  # answers are escaped
    assert "both facts stated" in page
    assert "GoogleAPIError: 503 UNAVAILABLE" in page
    assert '<a href="/evals/20261101T203045Z-baseline?only=failed">' in page


def test_eval_run_failures_only(eval_client: TestClient) -> None:
    page = eval_client.get("/evals/20261101T203045Z-baseline", params={"only": "failed"}).text

    assert "good-one" not in page
    assert "broken-one" in page
    assert "Showing failures only" in page


@pytest.mark.parametrize(
    "run_id", ["nope", "..%2F..%2Fapp%2Fconfig", "20261101T203045Z-baseline.json"]
)
def test_unknown_runs_are_not_found(eval_client: TestClient, run_id: str) -> None:
    assert eval_client.get(f"/evals/{run_id}").status_code == 404
