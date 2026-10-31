#!/bin/sh
# One-off setup for a native Postgres with pgvector, then the app role, database
# and vector extension. Safe to re-run. Use `make db-setup`, which adds sudo on Linux.
#
# Linux (Ubuntu/WSL): uses the installed Postgres; pgvector comes from the
#   PostgreSQL apt repository (PGDG). Run as root.
# macOS: Homebrew postgresql@17 + pgvector (brew's pgvector only builds for
#   postgresql@17 and @18), started with brew services. Run as your own user.
#
# Override the defaults with DB_NAME, DB_USER and DB_PASSWORD; keep DATABASE_URL
# in .env in step with them.

set -eu

DB_NAME="${DB_NAME:-mcp_rag}"
DB_USER="${DB_USER:-mcp_rag}"
DB_PASSWORD="${DB_PASSWORD:-mcp_rag}"
BREW_PG="postgresql@17"

setup_linux() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "error: on Linux run with sudo: sudo sh scripts/setup-postgres.sh"
    exit 1
  fi

  PG_MAJOR=$(ls /usr/lib/postgresql 2>/dev/null | sort -n | tail -1)
  if [ -z "$PG_MAJOR" ]; then
    echo "error: no Postgres server found. install it first: apt install postgresql"
    exit 1
  fi
  echo "postgres $PG_MAJOR found"

  if [ ! -f "/usr/share/postgresql/$PG_MAJOR/extension/vector.control" ]; then
    echo "installing pgvector from the PostgreSQL apt repository..."
    apt-get install -y postgresql-common
    /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
    apt-get install -y "postgresql-$PG_MAJOR-pgvector"
  fi

  PSQL="sudo -u postgres psql"
  ADMIN_DB="postgres"
}

setup_macos() {
  if [ "$(id -u)" -eq 0 ]; then
    echo "error: on macOS run without sudo; Homebrew refuses to run as root"
    exit 1
  fi
  if ! command -v brew > /dev/null 2>&1; then
    echo "error: Homebrew not found. install it: https://brew.sh"
    exit 1
  fi

  echo "installing $BREW_PG and pgvector with Homebrew (skipped if present)..."
  brew list "$BREW_PG" > /dev/null 2>&1 || brew install "$BREW_PG"
  brew list pgvector > /dev/null 2>&1 || brew install pgvector
  brew services start "$BREW_PG"

  PG_BIN="$(brew --prefix "$BREW_PG")/bin"
  echo "waiting for postgres to accept connections..."
  tries=0
  until "$PG_BIN/pg_isready" -q -h localhost; do
    tries=$((tries + 1))
    if [ "$tries" -ge 15 ]; then
      echo "error: postgres did not start. check: brew services info $BREW_PG"
      exit 1
    fi
    sleep 1
  done

  # Homebrew initialises the cluster with the current user as superuser.
  PSQL="$PG_BIN/psql -h localhost"
  ADMIN_DB="postgres"
}

case "$(uname -s)" in
  Linux) setup_linux ;;
  Darwin) setup_macos ;;
  *)
    echo "error: unsupported OS $(uname -s); install Postgres + pgvector manually"
    exit 1
    ;;
esac

echo "creating role and database (if missing)..."
$PSQL -v ON_ERROR_STOP=1 -d "$ADMIN_DB" \
  -v db="$DB_NAME" -v user="$DB_USER" -v pass="$DB_PASSWORD" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'user', :'pass')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'user') \gexec
SELECT format('CREATE DATABASE %I OWNER %I', :'db', :'user')
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'db') \gexec
SQL

echo "enabling the vector extension..."
$PSQL -v ON_ERROR_STOP=1 -d "$DB_NAME" -c 'CREATE EXTENSION IF NOT EXISTS vector'

echo "done. check the connection with: make db-check"
