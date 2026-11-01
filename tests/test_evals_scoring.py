from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from evals.questions import Question
from evals.scoring import Judgement, make_judge, outcome, score, summarise


def question(**overrides: Any) -> Question:
    values: dict[str, Any] = {
        "id": "q",
        "category": "version_specific",
        "question": "What is server/discover used for?",
        "spec_version": "2026-07-28",
        "expected": "answer",
        "key_facts": ["fact one", "fact two"],
        "sources": ["docs/a.mdx"],
    }
    return Question.model_validate({**values, **overrides})


def final(steps: list[str], route: str = "retrieve", **extra: Any) -> dict[str, Any]:
    trace = [{"step": s, "summary": "", "ms": 0, "tokens": 0, "details": []} for s in steps]
    return {"route": route, "trace": trace, "answer": "an answer", **extra}


def chunk(path: str) -> Document:
    return Document(page_content="x", metadata={"source_path": path})


def fake_judge(found: list[bool]) -> Any:
    calls: list[tuple[str, str]] = []

    def judge(q: Question, answer: str) -> tuple[list[bool], str, int]:
        calls.append((q.id, answer))
        return found, "because", 50

    judge.calls = calls  # type: ignore[attr-defined]
    return judge


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (final(["route", "decline"], route="out_of_scope"), "declined_out_of_scope"),
        (final(["route", "answer"], route="direct"), "direct"),
        (final(["route", "retrieve", "grade", "decline"]), "declined_not_covered"),
        (final(["route", "retrieve", "grade", "answer"]), "answered"),
        ({}, "answered"),
    ],
)
def test_outcome(state: dict[str, Any], expected: str) -> None:
    assert outcome(state) == expected


def test_answer_scores_retrieval_facts_and_citations() -> None:
    judge = fake_judge([True, False])
    state = final(
        ["route", "retrieve", "grade", "answer"],
        documents=[chunk("docs/other.mdx"), chunk("docs/a.mdx")],
        citations=[{"source_path": "docs/a.mdx"}],
    )

    result = score(question(), state, judge)

    assert result["retrieval_hit"] is True
    assert result["correctness"] == 0.5
    assert result["citation_hit"] is True
    assert result["passed"] is True  # half the facts meets PASS_FACT_SHARE
    assert (result["judge_reason"], result["judge_tokens"]) == ("because", 50)
    assert result["retrieved"] == ["docs/other.mdx", "docs/a.mdx"]


def test_answer_that_misses_sources_and_facts_fails() -> None:
    state = final(["route", "retrieve", "grade", "answer"], documents=[chunk("docs/x.mdx")])

    result = score(question(), state, fake_judge([False, False]))

    assert (result["retrieval_hit"], result["citation_hit"], result["passed"]) == (
        False,
        False,
        False,
    )


def test_false_decline_scores_zero_without_judging() -> None:
    judge = fake_judge([True, True])

    result = score(question(), final(["route", "retrieve", "grade", "decline"]), judge)

    assert result["correctness"] == 0.0
    assert result["passed"] is False
    assert judge.calls == []


@pytest.mark.parametrize(
    ("steps", "found", "passed", "judged"),
    [
        (["route", "retrieve", "grade", "decline"], [], True, False),
        (["route", "retrieve", "grade", "answer"], [True], True, True),
        (["route", "retrieve", "grade", "answer"], [False], False, True),
    ],
)
def test_not_covered_passes_on_decline_or_stated_absence(
    steps: list[str], found: list[bool], passed: bool, judged: bool
) -> None:
    q = question(
        category="absent_in_version", expected="not_covered", key_facts=["absent"], sources=[]
    )
    judge = fake_judge(found)

    result = score(q, final(steps), judge)

    assert result["passed"] is passed
    assert bool(judge.calls) is judged
    assert result["retrieval_hit"] is None


@pytest.mark.parametrize(
    ("category", "expected", "state", "passed"),
    [
        ("out_of_scope", "decline", final(["route", "decline"], route="out_of_scope"), True),
        ("out_of_scope", "decline", final(["route", "retrieve", "grade", "answer"]), False),
        ("direct", "direct", final(["route", "answer"], route="direct"), True),
        ("direct", "direct", final(["route", "retrieve", "grade", "answer"]), False),
    ],
)
def test_declines_and_direct_replies(
    category: str, expected: str, state: dict[str, Any], passed: bool
) -> None:
    q = question(category=category, expected=expected, key_facts=[], sources=[])

    assert score(q, state, fake_judge([]))["passed"] is passed


def judge_chain(parsed: Judgement | None, raw: AIMessage | None) -> RunnableLambda:
    seen: list[dict[str, str]] = []

    def reply(inputs: dict[str, str]) -> dict[str, Any]:
        seen.append(inputs)
        return {"raw": raw, "parsed": parsed}

    chain = RunnableLambda(reply)
    chain.seen = seen  # type: ignore[attr-defined]
    return chain


def test_make_judge_numbers_facts_and_pads_short_replies() -> None:
    usage = {"input_tokens": 9, "output_tokens": 1, "total_tokens": 10}
    raw = AIMessage(content="", usage_metadata=usage)
    chain = judge_chain(Judgement(facts=[True], reason="one found"), raw)

    found, reason, tokens = make_judge(chain)(question(), "an answer")

    assert chain.seen[0]["facts"] == "1. fact one\n2. fact two"  # type: ignore[attr-defined]
    assert chain.seen[0]["answer"] == "an answer"  # type: ignore[attr-defined]
    assert (found, reason, tokens) == ([True, False], "one found", 10)


def test_make_judge_handles_unparseable_and_missing_usage() -> None:
    found, reason, tokens = make_judge(judge_chain(None, None))(question(), "a")

    assert (found, reason, tokens) == ([False, False], "judge reply unparseable", 0)


def test_make_judge_truncates_long_replies() -> None:
    chain = judge_chain(Judgement(facts=[True, True, True], reason=""), AIMessage(content=""))

    assert make_judge(chain)(question(), "a")[0] == [True, True]


def result_row(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "error": None,
        "expected": "answer",
        "outcome": "answered",
        "passed": True,
        "retrieval_hit": True,
        "correctness": 1.0,
        "citation_hit": True,
        "seconds": 4.0,
        "tokens": 1000,
        "judge_tokens": 100,
    }
    return {**values, **overrides}


def test_summarise() -> None:
    results = [
        result_row(),
        result_row(passed=False, retrieval_hit=False, correctness=0.5, citation_hit=False),
        result_row(outcome="declined_not_covered", passed=False, correctness=0.0, seconds=2.0),
        result_row(expected="decline", outcome="declined_out_of_scope", seconds=1.0),
        result_row(expected="not_covered", outcome="answered", passed=False, seconds=3.0),
        result_row(expected="direct", outcome="direct", seconds=1.0),
        {"error": "boom", "passed": False},
    ]

    summary = summarise(results)

    assert summary["questions"] == 7
    assert summary["errors"] == 1
    assert summary["pass_rate"] == round(3 / 6, 3)
    assert summary["retrieval_hit_rate"] == round(2 / 3, 3)
    assert summary["correctness"] == 0.5
    assert summary["citation_accuracy"] == round(2 / 3, 3)
    assert summary["correct_declines"] == 0.5
    assert summary["false_declines"] == 1
    assert summary["direct_correct"] == 1.0
    assert summary["mean_seconds"] == round(15 / 6, 1)
    assert summary["total_tokens"] == 6 * 1100


def test_summarise_empty() -> None:
    summary = summarise([])

    assert summary["pass_rate"] is None
    assert summary["correctness"] is None
    assert summary["mean_seconds"] is None
