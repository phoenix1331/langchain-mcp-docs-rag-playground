import json
from pathlib import Path

import pytest

from ingest.navigation import SITE_URL, NavEntry, Navigation, page_key, page_url

INTRO = "docs/2026-07-28/getting-started/intro"
AUTH = "specification/2026-07-28/basic/authorization/index"

CONFIG = {
    "navigation": {
        "tabs": [
            {
                "tab": "Documentation",
                "versions": [
                    {
                        "version": "Version 2026-07-28 (latest)",
                        "pages": [
                            {"group": "Get started", "pages": [INTRO]},
                            {"group": "Examples", "pages": ["examples"]},
                        ],
                    },
                    {
                        "version": "Version 2025-11-25",
                        "pages": [{"group": "Examples", "pages": ["examples"]}],
                    },
                ],
            },
            {
                "tab": "Specification",
                "versions": [
                    {
                        "version": "Version 2026-07-28 (latest)",
                        "pages": [
                            "specification/2026-07-28/index",
                            {
                                "group": "Base Protocol",
                                "pages": [
                                    "specification/2026-07-28/basic/index",
                                    {
                                        "group": "Authorization",
                                        "pages": [AUTH],
                                    },
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                "tab": "Extensions",
                "pages": ["extensions/overview", {"pages": ["extensions/ungrouped"]}, 42],
            },
        ]
    }
}


@pytest.fixture
def nav() -> Navigation:
    return Navigation.from_config(CONFIG)


@pytest.mark.parametrize(
    ("source", "key"),
    [
        ("docs/specification/2026-07-28/basic/index.mdx", "specification/2026-07-28/basic/index"),
        (Path("docs/examples.mdx"), "examples"),
    ],
)
def test_page_key_strips_docs_dir_and_extension(source: str | Path, key: str) -> None:
    assert page_key(source) == key


@pytest.mark.parametrize(
    ("page", "url"),
    [
        ("index", SITE_URL),
        ("specification/2026-07-28/index", f"{SITE_URL}/specification/2026-07-28"),
        ("extensions/overview", f"{SITE_URL}/extensions/overview"),
        ("docs/2026-07-28/indexing", f"{SITE_URL}/docs/2026-07-28/indexing"),
    ],
)
def test_page_url(page: str, url: str) -> None:
    assert page_url(page) == url


def test_nested_groups_become_sections(nav: Navigation) -> None:
    entry = nav.lookup("docs/specification/2026-07-28/basic/authorization/index.mdx")

    assert entry.tab == "Specification"
    assert entry.sections == ("Base Protocol", "Authorization")
    assert entry.url == f"{SITE_URL}/specification/2026-07-28/basic/authorization"


def test_top_level_page_has_no_sections(nav: Navigation) -> None:
    entry = nav.lookup("docs/specification/2026-07-28/index.mdx")

    assert entry == NavEntry(page="specification/2026-07-28/index", tab="Specification")


def test_first_occurrence_wins_for_repeated_pages(nav: Navigation) -> None:
    entry = nav.lookup("docs/examples.mdx")

    assert entry.tab == "Documentation"
    assert entry.sections == ("Examples",)


def test_tabs_without_versions_and_groups_without_names(nav: Navigation) -> None:
    assert nav.lookup("docs/extensions/overview.mdx").tab == "Extensions"
    assert nav.lookup("docs/extensions/ungrouped.mdx").sections == ()
    assert len(nav) == 7


def test_unknown_page_falls_back_to_path_url(nav: Navigation) -> None:
    entry = nav.lookup("docs/extensions/hidden/page.mdx")

    assert entry == NavEntry(page="extensions/hidden/page")
    assert entry.url == f"{SITE_URL}/extensions/hidden/page"


def test_from_repo_reads_docs_json(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "docs.json").write_text(json.dumps(CONFIG))

    assert len(Navigation.from_repo(tmp_path)) == 7
