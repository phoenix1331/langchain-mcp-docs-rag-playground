import pytest

from app.retrieval.scope import ALL, in_scope, stored_versions, version_scope
from ingest.fetch import SpecVersions

VERSIONS = SpecVersions(("2025-06-18", "2025-11-25", "2026-07-28"))


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("latest", ("2026-07-28", "unversioned")),
        ("2025-06-18", ("2025-06-18", "unversioned")),
        (ALL, None),
    ],
)
def test_version_scope(version: str, expected: tuple[str, ...] | None) -> None:
    assert version_scope(version, VERSIONS) == expected


def test_unknown_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown spec version"):
        version_scope("2020-01-01", VERSIONS)


def test_in_scope() -> None:
    scope = ("2026-07-28", "unversioned")

    assert in_scope("unversioned", scope)
    assert not in_scope("2025-11-25", scope)
    assert in_scope("2025-11-25", None)


def test_stored_versions_are_sorted_and_skip_unversioned() -> None:
    versions = stored_versions({"2026-07-28", "unversioned", "2024-11-05"})

    assert versions.versions == ("2024-11-05", "2026-07-28")
    assert versions.latest == "2026-07-28"
