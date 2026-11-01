from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.retrieval.bm25 import SNAPSHOT_NAME, read_snapshot
from evals.questions import Category, Expected, Question, load_questions

QUESTIONS = load_questions()


def test_question_set_size_and_mix() -> None:
    counts = Counter(q.category for q in QUESTIONS)

    assert 30 <= len(QUESTIONS) <= 50
    assert counts[Category.VERSION_SPECIFIC] >= 15
    assert counts[Category.CROSS_VERSION] >= 5
    assert counts[Category.ABSENT_IN_VERSION] >= 3
    assert counts[Category.OUT_OF_SCOPE] >= 3


def test_every_spec_version_is_exercised() -> None:
    versions = {q.spec_version for q in QUESTIONS}

    assert {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25", "2026-07-28", "all"} <= versions


def test_sources_exist_in_the_current_index() -> None:
    """Catches typos and pages dropped upstream; needs a local ingest, so skips without one."""
    snapshot = get_settings().cache_dir / SNAPSHOT_NAME
    if not snapshot.exists():
        pytest.skip("no BM25 snapshot; run `make ingest`")
    indexed = {chunk.metadata["source_path"] for chunk in read_snapshot(snapshot)}

    missing = {(q.id, s) for q in QUESTIONS for s in q.sources if s not in indexed}

    assert missing == set()


def base(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "id": "x",
        "category": "version_specific",
        "question": "q",
        "spec_version": "2026-07-28",
        "expected": "answer",
        "key_facts": ["f"],
        "sources": ["docs/a.mdx"],
    }
    return {**values, **overrides}


def test_valid_question() -> None:
    assert Question.model_validate(base()).expected == Expected.ANSWER


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": "Not Kebab"}, "id"),
        ({"spec_version": "latest"}, "spec_version"),
        ({"expected": "decline"}, "version_specific questions expect answer"),
        ({"key_facts": []}, "key_facts are required"),
        ({"sources": []}, "sources are required"),
        (
            {"category": "out_of_scope", "expected": "decline", "key_facts": ["f"], "sources": []},
            "key_facts are required",
        ),
        (
            {"category": "absent_in_version", "expected": "not_covered", "sources": ["s"]},
            "sources are required",
        ),
    ],
)
def test_inconsistent_questions_are_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Question.model_validate(base(**overrides))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text(
        "- {id: a, category: direct, question: hi, spec_version: all, expected: direct,"
        " key_facts: [], sources: []}\n"
        "- {id: a, category: direct, question: hey, spec_version: all, expected: direct,"
        " key_facts: [], sources: []}\n"
    )

    with pytest.raises(ValueError, match="duplicate question ids: a"):
        load_questions(path)
