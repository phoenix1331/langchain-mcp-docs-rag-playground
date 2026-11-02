"""FastAPI app: the chat page, the question form and the answer stream.

`POST /ask` validates the question and returns a small fragment that opens a
server-sent event stream on `GET /ask/stream`. The stream runs the graph and sends:

- `stage`: a line as each graph step starts and finishes (route, search, grade, ...)
- `token`: answer text as the answer model writes it (HTML-escaped)
- `done`: the final rendered answer, or an error. Every stream ends with it, and the
  page closes the connection on it, so the browser never reconnects and re-asks.

Search resources and the compiled graph are built once at startup. Run with `make dev`.
"""

import html
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Protocol
from urllib.parse import urlencode

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt

from app.config import get_settings
from app.db import make_engine
from app.graph.build import build_question_graph
from app.graph.nodes import CITATION_PATTERN
from app.graph.progress import describe_result, describe_start
from app.retrieval.scope import ALL
from app.retrieval.search import Search
from evals.results import RESULTS_DIR, describe_config, format_stamp, load_run, load_runs
from ingest.fetch import SpecVersions

APP_DIR = Path(__file__).parent
MAX_QUESTION_CHARS = 1000
logger = logging.getLogger(__name__)


class QuestionGraph(Protocol):
    def astream(self, inputs: dict[str, str], stream_mode: list[str]) -> AsyncIterator[Any]: ...


DEFAULT_REPO_URL = "https://github.com/modelcontextprotocol/modelcontextprotocol"


@dataclass
class Services:
    versions: SpecVersions
    graph: QuestionGraph
    close: Callable[[], None] = field(default=lambda: None)
    repo_url: str = DEFAULT_REPO_URL  # source files link to `<repo_url>/blob/main/<path>`
    results_dir: Path = RESULTS_DIR  # saved eval runs for the evals page


def build_services() -> Services:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    search = Search.from_settings(settings, engine)
    return Services(
        versions=search.keyword.versions,
        graph=build_question_graph(settings, search),
        close=engine.dispose,
        repo_url=settings.mcp_repo_url.removesuffix(".git"),
    )


# Raw HTML in model output is escaped, so an answer can never inject markup.
_markdown = MarkdownIt("commonmark", {"html": False}).enable("table")


def render_markdown(text: str) -> str:
    return _markdown.render(text)


# Code spans and blocks are left alone: `[1]` there is code, not a citation.
_CODE = re.compile(r"(<pre>.*?</pre>|<code>.*?</code>)", re.S)


def link_citations(rendered: str, numbers: list[int]) -> str:
    """Turn `[n]`, `[n][m]` and `[n, m]` into links to `#source-n` for cited numbers."""
    known = set(numbers)

    def link(match: re.Match[str]) -> str:
        parts = [int(n) for n in match.group(1).split(",")]
        if not all(n in known for n in parts):
            return match.group(0)
        links = ", ".join(f'<a class="cite" href="#source-{n}">{n}</a>' for n in parts)
        return f"[{links}]"

    pieces = _CODE.split(rendered)
    return "".join(
        piece if index % 2 else CITATION_PATTERN.sub(link, piece)
        for index, piece in enumerate(pieces)
    )


def sse(event: str, data: str) -> str:
    """One server-sent event; every line of `data` gets its own `data:` field."""
    lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {event}\n{lines}\n"


def validation_error(question: str, version: str, versions: SpecVersions) -> str | None:
    if not question or len(question) > MAX_QUESTION_CHARS:
        return f"Ask a question of 1 to {MAX_QUESTION_CHARS} characters."
    if version != ALL and version not in versions.versions:
        return f"Unknown spec version {version!r}."
    return None


PROVIDER_ERROR = (
    "Something went wrong while answering. The model provider may be rate limiting "
    "requests; try again shortly."
)


def create_app(services_factory: Callable[[], Services] = build_services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = services_factory()
        app.state.services = services
        yield
        services.close()

    app = FastAPI(title="MCP docs assistant", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=APP_DIR / "templates")
    templates.env.filters["markdown"] = render_markdown
    templates.env.filters["link_citations"] = link_citations
    templates.env.filters["config"] = describe_config
    templates.env.filters["stamp"] = format_stamp
    templates.env.filters["pct"] = lambda value: "-" if value is None else f"{value:.0%}"

    def error(request: Request, message: str, status_code: int) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "_error.html", {"message": message}, status_code=status_code
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        versions: SpecVersions = request.app.state.services.versions
        context = {
            "versions": list(reversed(versions.versions)),
            "latest": versions.latest,
            "all": ALL,
            "max_chars": MAX_QUESTION_CHARS,
        }
        return templates.TemplateResponse(request, "index.html", context)

    @app.post("/ask", response_class=HTMLResponse)
    def ask(
        request: Request,
        question: Annotated[str, Form()],
        version: Annotated[str, Form()],
    ) -> HTMLResponse:
        """Validate, then return the fragment that opens the answer stream."""
        question = question.strip()
        problem = validation_error(question, version, request.app.state.services.versions)
        if problem:
            return error(request, problem, 422)
        stream_url = "/ask/stream?" + urlencode({"question": question, "version": version})
        context = {"question": question, "stream_url": stream_url}
        return templates.TemplateResponse(request, "_stream.html", context)

    @app.get("/ask/stream")
    async def ask_stream(request: Request, question: str, version: str) -> StreamingResponse:
        services: Services = request.app.state.services
        question = question.strip()

        def render(name: str, context: dict[str, Any]) -> str:
            return templates.get_template(name).render(context)

        async def events() -> AsyncIterator[str]:
            problem = validation_error(question, version, services.versions)
            if problem:
                yield sse("done", render("_error.html", {"message": problem}))
                return

            final: dict[str, Any] = {}
            inputs = {"question": question, "spec_version": version}
            try:
                async for mode, event in services.graph.astream(
                    inputs, stream_mode=["tasks", "messages", "values"]
                ):
                    if mode == "values":
                        final = event
                        continue
                    if mode == "messages":
                        chunk, meta = event
                        if meta.get("langgraph_node") == "answer" and chunk.text:
                            yield sse("token", html.escape(chunk.text))
                        continue
                    if "result" not in event:
                        text = describe_start(event["name"], event["input"])
                        yield sse("stage", f'<li class="stage">{html.escape(text)}</li>')
                        continue
                    text = describe_result(event["result"] or {}).strip()
                    yield sse("stage", f'<li class="stage-done">{html.escape(text)}</li>')
            except Exception:
                logger.exception("answering %r failed", question)
                yield sse("done", render("_error.html", {"message": PROVIDER_ERROR}))
                return

            trace = final.get("trace", [])
            context = {
                "result": final,
                "trace": trace,
                "citations": final.get("citations", []),
                "declined": bool(trace) and trace[-1]["step"] == "decline",
                "source_base": f"{services.repo_url}/blob/main/",
                "seconds": sum(step["ms"] for step in trace) / 1000,
                "tokens": sum(step["tokens"] for step in trace),
                "steps": len(trace),
            }
            yield sse("done", render("_answer_body.html", context))

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    @app.get("/evals", response_class=HTMLResponse)
    def evals_index(request: Request) -> HTMLResponse:
        runs = list(reversed(load_runs(request.app.state.services.results_dir)))
        return templates.TemplateResponse(request, "evals.html", {"runs": runs})

    @app.get("/evals/{run_id}", response_class=HTMLResponse)
    def eval_run(request: Request, run_id: str, only: str | None = None) -> HTMLResponse:
        run = load_run(run_id, request.app.state.services.results_dir)
        if run is None:
            raise HTTPException(status_code=404, detail="No such eval run")
        results = run.get("results", [])
        shown = [r for r in results if not r.get("passed")] if only == "failed" else results
        context = {"run": run, "run_id": run_id, "results": shown, "only": only}
        return templates.TemplateResponse(request, "eval_run.html", context)

    return app


app = create_app()
