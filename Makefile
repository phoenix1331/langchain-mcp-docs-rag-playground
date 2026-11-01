.DEFAULT_GOAL := help
PORT ?= 8000
.PHONY: help install db-setup db-check ingest query ask dev test coverage lint format

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies and git hooks
	@uv sync
	@git config core.hooksPath .githooks

db-setup: ## Install pgvector and create the database (sudo on Linux, Homebrew on macOS; safe to re-run)
	@if [ "$$(uname -s)" = "Darwin" ]; then sh scripts/setup-postgres.sh; else sudo sh scripts/setup-postgres.sh; fi

db-check: ## Check the database connection and pgvector
	@uv run db-check

ingest: ## Fetch the MCP docs and index new or changed chunks
	@uv run ingest

query: ## Inspect retrieval: make query Q="your question" [ARGS="--version all --mode bm25 --no-rerank"]
	@uv run query $(ARGS) "$(Q)"

ask: ## Answer a question through the graph: make ask Q="your question" [ARGS="--version all"]
	@uv run ask $(ARGS) "$(Q)"

dev: ## Run the web app with auto-reload (PORT, default 8000)
	@uv run uvicorn app.main:app --reload --port $(PORT)

test: ## Run the test suite
	@uv run pytest

coverage: ## Run tests with coverage (90% minimum on app/ and ingest/)
	@uv run pytest --cov --cov-report=term-missing --cov-fail-under=90

lint: ## Check linting and formatting without changing files
	@uv run ruff check .
	@uv run ruff format --check .

format: ## Auto-fix lint issues and format the code
	@uv run ruff check --fix .
	@uv run ruff format .
