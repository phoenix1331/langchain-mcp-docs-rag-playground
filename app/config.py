"""Typed application settings loaded from the environment and `.env`."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
MODEL_SPEC = r"^[a-z0-9_-]+:\S+$"  # provider:model, see app/models.py


class Settings(BaseSettings):
    """All runtime configuration. Models have no code defaults: they must come from config.

    Model settings are `provider:model` strings. Provider API keys are not settings: each
    LangChain integration reads its own variable (`GOOGLE_API_KEY`, `VOYAGE_API_KEY`, ...)
    from the environment, which `get_settings()` populates from `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "langchain-mcp-docs-rag-playground"

    database_url: str

    router_model: str = Field(pattern=MODEL_SPEC)
    answer_model: str = Field(pattern=MODEL_SPEC)
    embed_model: str = Field(pattern=MODEL_SPEC)
    embed_dimensions: int = Field(default=1024, ge=1)
    rerank_model: str = Field(pattern=MODEL_SPEC)

    # Embedding rate limits for ingestion (0 = unlimited) and request size in tokens.
    embed_rpm: int = Field(default=0, ge=0)
    embed_tpm: int = Field(default=0, ge=0)
    embed_batch_tokens: int = Field(default=100_000, ge=1)

    top_k: int = Field(default=20, ge=1)
    rerank_top_n: int = Field(default=6, ge=1)
    relevance_threshold: int = Field(default=2, ge=1)
    max_rewrites: int = Field(default=2, ge=0)

    chunk_size: int = Field(default=1500, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)

    mcp_repo_url: str = "https://github.com/modelcontextprotocol/modelcontextprotocol"
    cache_dir: Path = PROJECT_ROOT / ".cache"


@lru_cache
def get_settings() -> Settings:
    """Cached settings for the running process; also exports `.env` so integrations see keys."""
    load_dotenv(ENV_FILE, override=False)
    return Settings()
