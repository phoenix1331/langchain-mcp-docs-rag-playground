"""The eval question set: `evals/questions.yaml`, loaded and validated."""

from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator

QUESTIONS_FILE = Path(__file__).parent / "questions.yaml"
DATED_OR_ALL = r"^(\d{4}-\d{2}-\d{2}|all)$"


class Category(StrEnum):
    VERSION_SPECIFIC = "version_specific"
    CROSS_VERSION = "cross_version"
    ABSENT_IN_VERSION = "absent_in_version"
    OUT_OF_SCOPE = "out_of_scope"
    DIRECT = "direct"


class Expected(StrEnum):
    ANSWER = "answer"
    NOT_COVERED = "not_covered"
    DECLINE = "decline"
    DIRECT = "direct"


EXPECTED_FOR: dict[Category, Expected] = {
    Category.VERSION_SPECIFIC: Expected.ANSWER,
    Category.CROSS_VERSION: Expected.ANSWER,
    Category.ABSENT_IN_VERSION: Expected.NOT_COVERED,
    Category.OUT_OF_SCOPE: Expected.DECLINE,
    Category.DIRECT: Expected.DIRECT,
}


class Question(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    category: Category
    question: str = Field(min_length=1)
    spec_version: str = Field(pattern=DATED_OR_ALL)
    expected: Expected
    key_facts: list[str]
    sources: list[str]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        wanted = EXPECTED_FOR[self.category]
        if self.expected != wanted:
            raise ValueError(f"{self.id}: {self.category} questions expect {wanted}")
        needs_facts = self.expected in (Expected.ANSWER, Expected.NOT_COVERED)
        if needs_facts != bool(self.key_facts):
            raise ValueError(f"{self.id}: key_facts are required for answer and not_covered only")
        if (self.expected == Expected.ANSWER) != bool(self.sources):
            raise ValueError(f"{self.id}: sources are required for answer questions only")
        return self


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = [Question.model_validate(item) for item in raw]
    duplicates = [qid for qid, n in Counter(q.id for q in questions).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate question ids: {', '.join(duplicates)}")
    return questions
