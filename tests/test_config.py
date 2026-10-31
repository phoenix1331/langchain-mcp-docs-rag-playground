from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.config import MissingSecretError, Settings, get_settings, require_secret

REQUIRED = {
    "DATABASE_URL": "postgresql+psycopg://u:p@localhost:5432/db",
    "ROUTER_MODEL": "router-model",
    "ANSWER_MODEL": "answer-model",
    "EMBED_MODEL": "embed-model",
    "RERANK_MODEL": "rerank-model",
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key, value)
    return monkeypatch


def test_loads_required_values_and_defaults(env: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)

    assert settings.router_model == "router-model"
    assert settings.rerank_top_n == 6
    assert settings.max_rewrites == 2
    assert settings.google_api_key is None


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


def test_require_secret_returns_value() -> None:
    assert require_secret(SecretStr("abc"), "X_KEY") == "abc"


@pytest.mark.parametrize("value", [None, SecretStr("")])
def test_require_secret_names_missing_variable(value: SecretStr | None) -> None:
    with pytest.raises(MissingSecretError, match="VOYAGE_API_KEY"):
        require_secret(value, "VOYAGE_API_KEY")


def test_get_settings_is_cached(env: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()

    assert get_settings() is get_settings()
    get_settings.cache_clear()
