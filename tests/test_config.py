import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from app import config
from app.config import Settings, get_settings

REQUIRED = {
    "DATABASE_URL": "postgresql+psycopg://u:p@localhost:5432/db",
    "ROUTER_MODEL": "google_genai:router-model",
    "ANSWER_MODEL": "anthropic:answer-model",
    "EMBED_MODEL": "voyageai:embed-model",
    "RERANK_MODEL": "voyageai:rerank-model",
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key, value)
    return monkeypatch


def test_loads_required_values_and_defaults(env: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)

    assert settings.router_model == "google_genai:router-model"
    assert settings.rerank_top_n == 6
    assert settings.max_rewrites == 2


def test_thresholds_are_read_from_environment(env: pytest.MonkeyPatch) -> None:
    env.setenv("TOP_K", "40")
    env.setenv("MAX_REWRITES", "0")
    env.setenv("CACHE_DIR", "/tmp/somewhere")

    settings = Settings(_env_file=None)

    assert settings.top_k == 40
    assert settings.max_rewrites == 0
    assert settings.cache_dir == Path("/tmp/somewhere")


def test_model_names_have_no_code_defaults(env: pytest.MonkeyPatch) -> None:
    env.delenv("ANSWER_MODEL")

    with pytest.raises(ValidationError, match="answer_model"):
        Settings(_env_file=None)


def test_rejects_invalid_thresholds(env: pytest.MonkeyPatch) -> None:
    env.setenv("RERANK_TOP_N", "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("spec", ["gemini-3.6-flash", "Google:x", "google_genai:", ":model"])
def test_model_settings_must_be_provider_and_model(env: pytest.MonkeyPatch, spec: str) -> None:
    env.setenv("ROUTER_MODEL", spec)

    with pytest.raises(ValidationError, match="router_model"):
        Settings(_env_file=None)


def test_model_names_may_contain_colons(env: pytest.MonkeyPatch) -> None:
    env.setenv("ANSWER_MODEL", "ollama:llama3.2:3b")

    assert Settings(_env_file=None).answer_model == "ollama:llama3.2:3b"


def test_get_settings_exports_env_file_for_integrations(
    env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_PROVIDER_API_KEY=from-dotenv\n")
    env.setattr(config, "ENV_FILE", env_file)
    # Registering the variable with monkeypatch means it is removed again after the test.
    env.setenv("SOME_PROVIDER_API_KEY", "placeholder")
    env.delenv("SOME_PROVIDER_API_KEY")
    get_settings.cache_clear()

    get_settings()

    assert os.environ["SOME_PROVIDER_API_KEY"] == "from-dotenv"
    get_settings.cache_clear()


def test_get_settings_is_cached(env: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()

    assert get_settings() is get_settings()
    get_settings.cache_clear()
