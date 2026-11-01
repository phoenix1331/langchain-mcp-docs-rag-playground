import subprocess
from pathlib import Path

import pytest

from ingest.fetch import (
    SPEC_DIR,
    FetchError,
    SpecVersions,
    discover_versions,
    sync_repo,
)


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def commit_file(repo: Path, name: str, content: str) -> str:
    (repo / name).write_text(content)
    git("add", name, cwd=repo)
    git("commit", "-q", "-m", f"add {name}", cwd=repo)
    return git("rev-parse", "HEAD", cwd=repo)


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    repo = tmp_path / "upstream"
    repo.mkdir()
    git("init", "-q", "-b", "main", cwd=repo)
    git("config", "user.email", "test@example.com", cwd=repo)
    git("config", "user.name", "test", cwd=repo)
    git("config", "core.hooksPath", "/dev/null", cwd=repo)
    git("config", "commit.gpgsign", "false", cwd=repo)
    commit_file(repo, "a.mdx", "first")
    return repo


def make_spec(repo: Path, *names: str) -> Path:
    for name in names:
        (repo / SPEC_DIR / name).mkdir(parents=True)
    return repo


def test_sync_repo_clones_when_missing(upstream: Path, tmp_path: Path) -> None:
    dest = tmp_path / "cache" / "mcp"

    sha = sync_repo(upstream.as_uri(), dest)

    assert sha == git("rev-parse", "HEAD", cwd=upstream)
    assert (dest / "a.mdx").read_text() == "first"


def test_sync_repo_updates_existing_clone(upstream: Path, tmp_path: Path) -> None:
    dest = tmp_path / "mcp"
    sync_repo(upstream.as_uri(), dest)
    (dest / "a.mdx").write_text("local edit")
    new_sha = commit_file(upstream, "b.mdx", "second")

    sha = sync_repo(upstream.as_uri(), dest)

    assert sha == new_sha
    assert (dest / "b.mdx").read_text() == "second"
    assert (dest / "a.mdx").read_text() == "first"


def test_sync_repo_raises_fetch_error(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="git clone failed"):
        sync_repo((tmp_path / "missing").as_uri(), tmp_path / "dest")


def test_discover_versions_sorts_and_skips_non_dated(tmp_path: Path) -> None:
    repo = make_spec(tmp_path, "2025-06-18", "draft", "2024-11-05", "2026-07-28")
    (repo / SPEC_DIR / "2099-01-01.mdx").touch()

    found = discover_versions(repo)

    assert found.versions == ("2024-11-05", "2025-06-18", "2026-07-28")
    assert found.latest == "2026-07-28"


def test_discover_versions_requires_spec_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_versions(tmp_path)


def test_discover_versions_requires_a_dated_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no dated spec versions"):
        discover_versions(make_spec(tmp_path, "draft"))


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("latest", "2025-06-18"), ("2024-11-05", "2024-11-05")],
)
def test_resolve_maps_latest(requested: str, expected: str) -> None:
    versions = SpecVersions(versions=("2024-11-05", "2025-06-18"))

    assert versions.resolve(requested) == expected


def test_resolve_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="draft"):
        SpecVersions(versions=("2025-06-18",)).resolve("draft")
