# MCP Docs RAG Playground

Ask questions about the Model Context Protocol (MCP) specification and get answers tied to the spec release you choose, cited back to the exact sections, from an agentic RAG pipeline built with LangChain and LangGraph.

![Asking a question: the answer streams in with numbered citations and a sources panel](docs/screenshots/answer.png)

## What this demonstrates

- **Version-aware retrieval.** The MCP spec ships as dated releases (2024-11-05 to 2026-07-28). Every chunk is tagged with its release, and a question only searches the release you pick plus version-independent pages, so answers do not mix what one version says with another.
- **An agentic question graph in LangGraph.** A router decides whether a question needs the docs, a grader checks the retrieved chunks, a rewrite loop tries a better query when too few are relevant, and a decline path says plainly when the docs do not cover something. Each step is a node; the routing is conditional edges.
- **Hybrid search with reranking.** BM25 keyword search and pgvector semantic search are merged with reciprocal rank fusion, then Voyage reranks the candidates down to the six the answer sees.
- **Incremental ingestion.** The LangChain indexing API with a SQL record manager skips unchanged chunks and deletes removed ones, and an embedding cache means text repeated across spec releases is embedded once.
- **A UI that shows its working.** Answers stream in over server-sent events while each graph step reports what it is doing; a "How I got this answer" panel lists the route decision, every retrieved chunk with its scores, the grades, any rewritten query, and the time and token cost of each step.
- **Measured quality.** A fixed set of 46 questions, checked against the source docs, scores retrieval, answer correctness, citations and declines for each configuration, so changes are judged by numbers rather than by eye.
- **An honest look at the frameworks.** Where LangChain and LangGraph earned their place, and where a few lines of plain code did better (see [below](#where-langchain-and-langgraph-help-and-where-plain-code-wins)).

## How it works

**Ingestion** (`make ingest`) turns the docs into searchable chunks:

```mermaid
flowchart LR
    A["MCP docs repo<br/>MDX + docs.json"] --> B["Clean MDX<br/>keep code and tables"]
    B --> C["Split by headings,<br/>then by size"]
    C --> D["Embed with Voyage<br/>cached, rate limited"]
    D --> E[("pgvector")]
    E --> F[("BM25 snapshot")]
```

**The question graph** (LangGraph) decides how each question is handled:

```mermaid
flowchart TD
    START([question + spec release]) --> route{route}
    route -->|needs the docs| retrieve["retrieve<br/>BM25 + vector, fused, reranked"]
    route -->|greeting or help| answer
    route -->|off topic| decline
    stores[("pgvector +<br/>BM25 snapshot")] -.-> retrieve
    retrieve --> grade{grade}
    grade -->|enough relevant chunks| answer["answer<br/>cited, relevant chunks only"]
    grade -->|too few| rewrite[rewrite the query]
    rewrite --> retrieve
    grade -->|still too few after 2 rewrites| decline["decline<br/>with the closest sections"]
    answer --> END([streamed answer + trace])
    decline --> END
```

1. **Ingest** clones the MCP docs repo, cleans the Mintlify MDX (components unwrapped, code blocks and tables kept), maps each page to its site URL and navigation section, and splits pages on their headings. Each chunk carries its heading path (`Base Protocol › Authorization › Refresh Tokens`) so short sections keep their context.
2. **Route** asks a fast model whether the question needs the docs, is small talk, or is off topic.
3. **Retrieve** runs BM25 and vector search over the chosen release, merges the two rankings and reranks them.
4. **Grade** checks all retrieved chunks in one call and notes what is missing. Too few relevant chunks sends the graph to **rewrite**, which rephrases the search using what was missing, up to twice; after that it **declines** and lists the closest sections it found.
5. **Answer** writes from the relevant chunks only, citing each claim. Sources are labelled Specification, Guide or Extension, because only the specification defines what a release supports (see [architecture notes](#architecture-notes)).

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.12+, managed with `uv` |
| Orchestration | LangGraph for the question graph; LangChain for loaders, splitters, retrievers, the indexing API and model wrappers |
| Chat models | Gemini 3.1 Flash-Lite for routing, grading, rewriting and answers (used because it's free; larger models will return better results). Any LangChain chat integration can be swapped in |
| Embeddings and rerank | Voyage `voyage-4` and `rerank-2.5` on the free tier (used because they're free; larger models will return better results) |
| Store | Postgres with pgvector, running natively (no Docker) |
| Keyword search | BM25 via `rank-bm25`, behind a small LangChain retriever |
| Web | FastAPI, Jinja templates, HTMX with server-sent events; no front-end build step |
| Evals | LLM-as-judge scoring, JSON results per run, optional LangSmith experiments |
| Tooling | `ruff`, `pytest` with 100% line coverage on `app/` and `ingest/`, git hooks with `osv-scanner` |

## Getting started

### Prerequisites

- **Python 3.12+ and [uv](https://docs.astral.sh/uv/)**.
- **Postgres with pgvector.** `make db-setup` installs pgvector and creates the database: on Linux and WSL it uses the PGDG apt packages (needs `sudo`), on macOS Homebrew `postgresql@17`. It expects Postgres on port 5432.
- **API keys.** A Google AI Studio key for Gemini and a [Voyage AI](https://www.voyageai.com) key. Both free tiers work, with limits (see [running on free tiers](#running-on-free-tiers)). Copy `.env.example` to `.env` and fill in `GOOGLE_API_KEY` and `VOYAGE_API_KEY`. A `LANGSMITH_API_KEY` is optional.
- **[osv-scanner](https://github.com/google/osv-scanner)** for the pre-commit security audit: `go install github.com/google/osv-scanner/cmd/osv-scanner@latest`.
- `make install` must be run once to wire the git hooks in `.githooks/` (commit message format, `ruff`, `osv-scanner`).

### Steps

1. `make install` installs dependencies and the git hooks
2. `make db-setup` installs pgvector and creates the database
3. `make db-check` confirms Postgres and pgvector are reachable
4. `make ingest` fetches the MCP docs and indexes them; later runs only embed what changed
5. `make dev` starts the app on http://localhost:8000 (set `PORT` to use another)
6. `make ask Q="How does a client refresh an OAuth access token?"` answers from the terminal, showing each step

### Commands

| Command | Does |
|---|---|
| `make query Q="..." ARGS="--version all --mode bm25 --no-rerank"` | Show what retrieval returns, with scores, without calling a chat model |
| `make ask Q="..." ARGS="--version 2025-06-18"` | Answer through the full graph; progress on stderr, answer on stdout |
| `make evals ARGS="--label baseline"` | Run the eval questions (`--ids`, `--limit`, `--resume FILE`, `--mode`, `--no-rerank`, `--no-grading`) |
| `make evals-compare` | Compare every saved eval run side by side |
| `make test` / `make coverage` | Run the tests / with coverage (90% minimum) |
| `make lint` / `make format` | Check / fix style with `ruff` |
| `make help` | List every target |

### Running on free tiers

Free tiers shape how this runs, and the defaults in `.env.example` are set for them:

- **Voyage** without a payment method advertises 3 requests and 10K tokens per minute but rejected batches much over 1,500 tokens in practice, so ingestion batches at 1,500 tokens and 3,000 per minute. The first ingest of about 4,700 chunks takes over an hour; re-runs take seconds because unchanged chunks are skipped and repeated text comes from the embedding cache.
- **Gemini** free keys have a daily request limit per model: some Flash models stopped at 20 a day, while Flash-Lite handled nearly 300 a day during these eval runs without hitting one. A question costs about three calls (route, grade, answer).
- **Evals** pause 40 seconds between questions (`EVAL_PAUSE_SECONDS`) and retry rate limits and overloaded-model errors; a daily quota stops a run cleanly and `--resume` continues it later. A full run of 46 questions takes about an hour.

## Environment variables

| Key | Default in `.env.example` | Description |
|---|---|---|
| `GOOGLE_API_KEY`, `VOYAGE_API_KEY` | | Keys for the example providers. Each LangChain integration reads its own variable, so another provider needs its own key (e.g. `ANTHROPIC_API_KEY`) |
| `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING` | | Optional tracing, and the dataset and experiment for each eval run |
| `DATABASE_URL` | `postgresql+psycopg://mcp_rag:mcp_rag@localhost:5432/mcp_rag` | Postgres with pgvector |
| `ROUTER_MODEL` | `google_genai:gemini-3.1-flash-lite` | Fast model for routing, grading and rewriting, as `provider:model` (used because it's free; larger models will return better results) |
| `ANSWER_MODEL` | `google_genai:gemini-3.1-flash-lite` | Model that writes answers (used because it's free; a larger model will return better results) |
| `JUDGE_MODEL` | blank | Eval judge; blank uses `ANSWER_MODEL`. A different provider avoids a model grading its own answers |
| `EMBED_MODEL` | `voyageai:voyage-4` | Embedding model. Changing it indexes into a new collection |
| `EMBED_DIMENSIONS` | `1024` | Vector size; must match the embedding model |
| `EMBED_RPM`, `EMBED_TPM`, `EMBED_BATCH_TOKENS` | `2`, `3000`, `1500` | Ingestion rate limits and batch size (0 = unlimited) |
| `RERANK_MODEL` | `voyageai:rerank-2.5` | Reranker; `voyageai` and `cohere` are supported |
| `TOP_K` | `20` | Candidates from each retriever before reranking |
| `RERANK_TOP_N` | `6` | Chunks the answer sees |
| `RELEVANCE_THRESHOLD` | `2` | Relevant chunks needed to answer rather than rewrite |
| `MAX_REWRITES` | `2` | Query rewrites before declining |
| `EVAL_PAUSE_SECONDS` | `40` | Pause between eval questions for free-tier quotas |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `1500`, `150` | Chunk size and overlap in characters |
| `MCP_REPO_URL` | the MCP docs repo | Source of the docs |
| `CACHE_DIR` | `.cache` | Repo clone, embedding cache and BM25 snapshot (git-ignored) |

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | Ask page: question box and spec release picker |
| `POST /ask` | Validates a question and returns the fragment that opens the answer stream |
| `GET /ask/stream?question=...&version=...` | Server-sent events: `stage` per graph step, `token` as the answer is written, `done` with the rendered answer |
| `GET /evals` | Every saved eval run with its configuration and scores |
| `GET /evals/{run}?only=failed` | One run's per-question results |
| `GET /health` | Liveness check |

`request.http` has these for the VS Code REST Client and JetBrains HTTP Client.

<p align="center">
  <img src="docs/screenshots/trace.png" width="480" alt="The How I got this answer panel: route decision, retrieved chunks with scores, grades and costs">
</p>

## Evaluation

`evals/questions.yaml` holds 46 questions, each checked against the docs repo: version-specific questions for every release, cross-version questions ("when was batching added, and when was it removed?"), questions about features a release does not have, off-topic questions that should be declined, and greetings. Each has the key facts a correct answer must state and the source files those facts come from.

`make evals` runs every question through the real graph and scores:

| Metric | Meaning |
|---|---|
| Pass | The expected behaviour happened and, for answers, at least half the key facts were stated |
| Retrieval | An expected source file was among the chunks the answer saw |
| Correct | Share of key facts stated, judged by an LLM |
| Citations | The answer cited at least one expected source |
| Declines | Off-topic and not-covered questions were declined, or answered with "this release does not define it" |

### Results

<!-- EVAL_RESULTS -->

![Eval runs with their configuration and scores](docs/screenshots/evals.png)

## Architecture notes

- **Versioned guides are a trap.** The docs site keeps a copy of its guides for every release, but they are near-copies of the current guides with links rewritten: the 2024-11-05 guide describes OAuth flows that the 2024-11-05 specification does not have. So sources are labelled Specification, Guide or Extension, and the answer prompt says only the specification defines what a release supports. Asked about authorization for 2024-11-05, the assistant now says the spec does not define it.
- **One collection per embedding model.** The indexing API only re-embeds changed text, so switching embedding models in place would keep the old vectors and search would quietly mix two vector spaces. Each model and size gets its own collection, and chunk ids include the collection, because pgvector stores every collection in one table keyed by id.
- **Batched grading.** Grading each chunk separately would mean six model calls per search on a free tier; one structured call grades them all and says what is missing, which feeds the rewrite.
- **Streams never reconnect on their own.** The browser's EventSource reconnects after errors, which here would re-run the whole graph and spend quota. Every stream ends with a `done` event, even for errors, and the page closes the connection when it arrives.
- **Models are configuration.** Every model setting is `provider:model`, built through LangChain's `init_chat_model` and `init_embeddings`, so Gemini and Voyage are examples rather than dependencies.
- **Evals pin dated releases.** Questions use a dated release or `all`, never `latest`, so the next spec release does not change what the eval measures.

## Where LangChain and LangGraph help, and where plain code wins

| Piece | Verdict |
|---|---|
| Question graph | **LangGraph helps.** Conditional edges, the capped rewrite loop and a streamed event per node map directly onto the brief, and the task stream drives the live progress in the UI |
| Indexing API and record manager | **LangChain helps.** Skipping unchanged chunks and deleting removed pages comes for free; one surprise (shared ids across collections) needed a custom key |
| Splitters | **LangChain helps**, with care: the Markdown splitter needed its pieces merged back per heading to keep code blocks with their text |
| Structured output | **LangChain helps.** `with_structured_output(..., include_raw=True)` gives typed decisions and the token counts for the trace panel |
| BM25 | **Plain code.** `BM25Retriever` lives in the sunset `langchain-community` package and is a thin wrapper; a short retriever over `rank-bm25` also fixed its IDF, which gave common words like "server" almost no weight |
| Hybrid fusion | **Plain code.** `EnsembleRetriever` merges results with identical text, which collapses the same section from different releases into one; fusing on chunk id keeps them apart |
| Reranking | **Mostly LangChain**, subclassed: `VoyageAIRerank` drops document ids that citations need |
| MDX loading | **Plain code.** Mintlify components and generated schema pages needed bespoke cleaning that no generic loader handles |
| Ingestion pipeline | **Plain code.** It is a straight line, so a graph would add nothing |

## Compared with laravel-docs-rag-playground

Both projects answer questions from a docs site with citations; this one exists partly to compare a framework-built pipeline with that hand-built one.

| | [laravel-docs-rag-playground](https://github.com/phoenix1331/laravel-docs-rag-playground) | This project |
|---|---|---|
| Approach | A hand-built RAG pipeline in Laravel | An agentic graph in LangGraph |
| Model calls per question | 1 | About 3 (route, grade, answer); up to 6 with rewrites |
| Not-covered questions | A similarity threshold refuses clear misses without a model call | A grader decides, and a rewrite loop tries again first |
| Search | Vector by default; hybrid measured lower on that data | Hybrid plus a reranker; hybrid measured slightly higher here |
| Evaluation | Retrieval only (recall@k, MRR) | Retrieval, LLM-judged correctness, citations and declines |

The extra model calls buy routing, version honesty and a second attempt at retrieval, at the cost of latency and quota. Both projects found that measuring beat assumptions, and their hybrid results point in opposite directions on small question sets, which is a reason to measure on your own data.

## Running ingestion on a schedule

Ingestion is manual today, but it is built to be re-run: unchanged chunks are skipped and deleted pages are removed. Two easy ways to automate it:

- **On a schedule:** a GitHub Actions workflow on a `cron` trigger that runs `make ingest` against a hosted Postgres, with the API keys as repository secrets. The embedding cache can be stored with `actions/cache` so repeated text is not embedded again.
- **On new spec commits:** a small workflow in a fork of the MCP docs repo, or a webhook, that sends a `repository_dispatch` event to this repo on each push to `main`, triggering the same ingest job. New spec releases appear as new dated folders and are picked up automatically.

## Credits and licence

The answers come from the [Model Context Protocol documentation](https://modelcontextprotocol.io) by the MCP contributors, fetched at run time from [modelcontextprotocol/modelcontextprotocol](https://github.com/modelcontextprotocol/modelcontextprotocol) and not redistributed in this repo. The MCP project is moving from MIT to Apache-2.0: the specification is licensed under Apache-2.0, the documentation under CC BY 4.0, and some older contributions remain under MIT ([licence](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/LICENSE)).

This is an independent learning project, not affiliated with or endorsed by the Model Context Protocol project.
