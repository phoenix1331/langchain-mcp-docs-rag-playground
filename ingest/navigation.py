"""Map source files to site URLs and navigation sections using Mintlify's `docs.json`."""

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ingest.fetch import DOCS_DIR, NAV_CONFIG

SITE_URL = "https://modelcontextprotocol.io"


@dataclass(frozen=True)
class NavEntry:
    """Where a page sits on the site: its tab and the nested group names leading to it."""

    page: str
    tab: str = ""
    sections: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        return page_url(self.page)


def page_key(source_path: str | Path) -> str:
    """`docs/specification/v/basic/index.mdx` becomes `specification/v/basic/index`."""
    path = PurePosixPath(Path(source_path).as_posix())
    return path.relative_to(DOCS_DIR.as_posix()).with_suffix("").as_posix()


def page_url(page: str) -> str:
    """Canonical site URL for a page key; `index` pages resolve to their folder."""
    if page == "index":
        return SITE_URL
    return f"{SITE_URL}/{page.removesuffix('/index')}"


class Navigation:
    """Lookup from page key to `NavEntry`. The first occurrence of a page wins."""

    def __init__(self, entries: dict[str, NavEntry]) -> None:
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Navigation":
        entries: dict[str, NavEntry] = {}
        for tab in config.get("navigation", {}).get("tabs", []):
            name = tab.get("tab", "")
            for version in tab.get("versions", []):
                _collect(version.get("pages", []), name, (), entries)
            _collect(tab.get("pages", []), name, (), entries)
        return cls(entries)

    @classmethod
    def from_repo(cls, repo: Path) -> "Navigation":
        return cls.from_config(json.loads((repo / NAV_CONFIG).read_text(encoding="utf-8")))

    def lookup(self, source_path: str | Path) -> NavEntry:
        """Entry for a source file; pages missing from the nav get a URL from their path."""
        key = page_key(source_path)
        return self._entries.get(key) or NavEntry(page=key)


def _collect(
    pages: list[Any], tab: str, sections: tuple[str, ...], entries: dict[str, NavEntry]
) -> None:
    for item in pages:
        if isinstance(item, str):
            entries.setdefault(item, NavEntry(page=item, tab=tab, sections=sections))
            continue
        if isinstance(item, dict):
            group = item.get("group")
            nested = (*sections, group) if group else sections
            _collect(item.get("pages", []), tab, nested, entries)
