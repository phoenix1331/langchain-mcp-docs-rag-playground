"""Database engine factory and the `db-check` connection check."""

import sys
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings


def make_engine(url: str | None = None) -> Engine:
    """SQLAlchemy engine for the configured `DATABASE_URL`."""
    return create_engine(url or get_settings().database_url, pool_pre_ping=True)


@dataclass(frozen=True)
class DbStatus:
    server_version: str
    pgvector_version: str | None

    @property
    def ready(self) -> bool:
        return self.pgvector_version is not None


def check_database(engine: Engine) -> DbStatus:
    """Connect and report the server version and installed pgvector version, if any."""
    with engine.connect() as conn:
        server_version = conn.execute(text("SHOW server_version")).scalar_one()
        pgvector_version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
    return DbStatus(server_version=server_version, pgvector_version=pgvector_version)


def main() -> int:
    """Entry point for `uv run db-check`."""
    engine = make_engine()
    try:
        status = check_database(engine)
    except SQLAlchemyError as exc:
        print(f"error: cannot connect to {engine.url.render_as_string()}", file=sys.stderr)
        print(f"  {exc.__cause__ or exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    print(f"connected: postgres {status.server_version}")
    if not status.ready:
        print("error: pgvector extension is not enabled in this database", file=sys.stderr)
        print("  run: sudo sh scripts/setup-postgres.sh", file=sys.stderr)
        return 1

    print(f"pgvector {status.pgvector_version} enabled")
    return 0
