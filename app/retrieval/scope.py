"""Which stored chunks a question may search, given the selected spec version."""

from ingest.fetch import UNVERSIONED, SpecVersions

ALL = "all"

Scope = tuple[str, ...] | None
"""`spec_version` values to search; None searches every chunk."""


def version_scope(version: str, versions: SpecVersions) -> Scope:
    """`all` searches everything; `latest` or a dated version also includes unversioned pages."""
    if version == ALL:
        return None
    return (versions.resolve(version), UNVERSIONED)


def in_scope(spec_version: str, scope: Scope) -> bool:
    return scope is None or spec_version in scope


def stored_versions(spec_versions: set[str]) -> SpecVersions:
    """Dated versions present in the index, oldest first. Dates sort as strings."""
    return SpecVersions(tuple(sorted(spec_versions - {UNVERSIONED})))
