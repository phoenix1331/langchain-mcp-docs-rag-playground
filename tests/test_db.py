from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from app import db
from app.db import DbStatus, check_database, make_engine


def fake_engine(server_version: str = "14.24", pgvector: str | None = "0.8.1") -> MagicMock:
    conn = MagicMock()
    conn.execute.side_effect = [
        MagicMock(scalar_one=MagicMock(return_value=server_version)),
        MagicMock(scalar_one_or_none=MagicMock(return_value=pgvector)),
    ]
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    engine.url = create_engine("postgresql+psycopg://u:secret@localhost/db").url
    return engine


def test_make_engine_uses_given_url() -> None:
    engine = make_engine("postgresql+psycopg://u:p@localhost:5432/example")

    assert engine.url.database == "example"
    assert engine.url.drivername == "postgresql+psycopg"


def test_check_database_reports_versions() -> None:
    status = check_database(fake_engine())

    assert status == DbStatus(server_version="14.24", pgvector_version="0.8.1")
    assert status.ready


def test_status_not_ready_without_pgvector() -> None:
    assert not check_database(fake_engine(pgvector=None)).ready


def test_main_succeeds_when_pgvector_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(db, "make_engine", lambda: fake_engine())

    assert db.main() == 0
    assert "pgvector 0.8.1 enabled" in capsys.readouterr().out


def test_main_fails_without_pgvector(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(db, "make_engine", lambda: fake_engine(pgvector=None))

    assert db.main() == 1
    assert "setup-postgres.sh" in capsys.readouterr().err


def test_main_fails_cleanly_when_unreachable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = fake_engine()
    engine.connect.side_effect = OperationalError("connect", {}, Exception("refused"))
    monkeypatch.setattr(db, "make_engine", lambda: engine)

    assert db.main() == 1
    err = capsys.readouterr().err
    assert "cannot connect" in err
    assert "secret" not in err
    engine.dispose.assert_called_once()


@pytest.mark.db
def test_live_database_has_pgvector() -> None:
    engine = make_engine()
    try:
        status = check_database(engine)
    except OperationalError:
        pytest.skip("database not reachable")
    finally:
        engine.dispose()

    assert status.ready
