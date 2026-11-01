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
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Protocol
from urllib.parse import urlencode

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt

from app.config import get_settings
from app.db import make_engine
from app.graph.build import build_question_graph
from app.graph.progress import describe_result, describe_start
from app.retrieval.scope import ALL
from app.retrieval.search import Search
from ingest.fetch import SpecVersions

APP_DIR = Path(__file__).parent
MAX_QUESTION_CHARS = 1000
logger = logging.getLogger(__name__)


class QuestionGraph(Protocol):
    def astream(self, inputs: dict[str, str], stream_mode: list[str]) -> AsyncIterator[Any]: ...


@dataclass
class Services:
    versions: SpecVersions
    graph: QuestionGraph
    close: Callable[[], None] = field(default=lambda: None)


def build_services() -> Services:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    search = Search.from_settings(settings, engine)
    return Services(
        versions=search.keyword.versions,
        graph=build_question_graph(settings, search),
        close=engine.dispose,
    )


# Raw HTML in model output is escaped, so an answer can never inject markup.
_markdown = MarkdownIt("commonmark", {"html": False}).enable("table")


def render_markdown(text: str) -> str:
    return _markdown.render(text)


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
                "seconds": sum(step["ms"] for step in trace) / 1000,
                "tokens": sum(step["tokens"] for step in trace),
                "steps": len(trace),
            }
            yield sse("done", render("_answer_body.html", context))

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    return app


app = create_app()
