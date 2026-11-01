"""FastAPI app: the chat page and the question endpoint.

Search resources and the compiled graph are built once at startup and shared by every
request. Run with `make dev`.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Protocol

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt

from app.config import get_settings
from app.db import make_engine
from app.graph.build import build_question_graph
from app.retrieval.scope import ALL
from app.retrieval.search import Search
from ingest.fetch import SpecVersions

APP_DIR = Path(__file__).parent
MAX_QUESTION_CHARS = 1000
logger = logging.getLogger(__name__)


class QuestionGraph(Protocol):
    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]: ...


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
        services: Services = request.app.state.services
        question = question.strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            return error(request, f"Ask a question of 1 to {MAX_QUESTION_CHARS} characters.", 422)
        if version != ALL and version not in services.versions.versions:
            return error(request, f"Unknown spec version {version!r}.", 422)

        try:
            result = services.graph.invoke({"question": question, "spec_version": version})
        except Exception:
            # Provider failures (rate limits, timeouts) surface here; the page stays usable.
            logger.exception("answering %r failed", question)
            return error(
                request,
                "Something went wrong while answering. The model provider may be rate "
                "limiting requests; try again shortly.",
                502,
            )

        trace = result.get("trace", [])
        context = {
            "question": question,
            "result": result,
            "seconds": sum(step["ms"] for step in trace) / 1000,
            "tokens": sum(step["tokens"] for step in trace),
            "steps": len(trace),
        }
        return templates.TemplateResponse(request, "_answer.html", context)

    return app


app = create_app()
