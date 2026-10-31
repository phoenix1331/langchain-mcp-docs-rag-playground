"""Fetch the MCP docs repo into the local cache and discover spec versions."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path("docs")
SPEC_DIR = DOCS_DIR / "specification"
GUIDES_DIR = DOCS_DIR / "docs"
NAV_CONFIG = DOCS_DIR / "docs.json"

LATEST = "latest"
UNVERSIONED = "unversioned"

_VERSION_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class FetchError(RuntimeError):
    """Raised when the repo cannot be cloned or updated."""


@dataclass(frozen=True)
class SpecVersions:
    """Dated spec versions, oldest first. `draft` and other non-dated folders are excluded."""

    versions: tuple[str, ...]

    @property
    def latest(self) -> str:
        return self.versions[-1]

    def resolve(self, version: str) -> str:
        """Map `latest` to the newest dated version; reject unknown versions."""
        resolved = self.latest if version == LATEST else version
        if resolved not in self.versions:
            raise ValueError(f"unknown spec version {version!r}; known: {', '.join(self.versions)}")
        return resolved


def _git(*args: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise FetchError(f"git {args[0]} failed: {exc.stderr.strip()}") from exc
    return result.stdout.strip()


def sync_repo(url: str, dest: Path) -> str:
    """Shallow-clone `url` into `dest`, or fast-forward an existing clone. Returns the HEAD sha."""
    if (dest / ".git").is_dir():
        _git("fetch", "--depth", "1", "origin", "HEAD", cwd=dest)
        _git("reset", "--hard", "FETCH_HEAD", cwd=dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--depth", "1", url, str(dest))
    return _git("rev-parse", "HEAD", cwd=dest)


def discover_versions(repo: Path) -> SpecVersions:
    """Find dated version folders under the spec directory."""
    spec_dir = repo / SPEC_DIR
    if not spec_dir.is_dir():
        raise FileNotFoundError(f"spec directory not found: {spec_dir}")

    versions = sorted(
        child.name
        for child in spec_dir.iterdir()
        if child.is_dir() and _VERSION_PATTERN.match(child.name)
    )
    if not versions:
        raise ValueError(f"no dated spec versions found in {spec_dir}")
    return SpecVersions(versions=tuple(versions))
