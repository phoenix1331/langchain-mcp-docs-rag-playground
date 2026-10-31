"""Typed application settings loaded from the environment and `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All runtime configuration. Model names have no code defaults: they must come from config."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys are optional at load time so commands that do not need them (e.g. db-check)
    # still run. Code that calls a provider checks for its key via `require_secret`.
    google_api_key: SecretStr | None = None
    voyage_api_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "langchain-mcp-docs-rag-playground"

    database_url: str

    router_model: str
    answer_model: str
    embed_model: str
    embed_dimensions: int = Field(default=1024, ge=1)
    rerank_model: str

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


class MissingSecretError(RuntimeError):
    """Raised when a provider is used without its API key configured."""


def require_secret(value: SecretStr | None, env_name: str) -> str:
    """Return the secret's plain value, or fail with a message naming the missing variable."""
    if value is None or not value.get_secret_value():
        raise MissingSecretError(f"{env_name} is not set; add it to .env")
    return value.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance for the running process."""
    return Settings()
