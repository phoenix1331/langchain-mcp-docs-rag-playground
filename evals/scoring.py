"""Scoring one graph result against its eval question, and summarising a run.

Metrics, per the brief:

- retrieval hit: an expected source file is among the final retrieved chunks
- correctness: the share of key facts an LLM judge finds in the answer
- citation accuracy: the answer cites at least one expected source file
- correct declines: out-of-scope questions declined, absent-in-version questions
  declined or answered with an explicit "this version does not define it"

A question passes when its expected behaviour happened and, for answers, at least
`PASS_FACT_SHARE` of its key facts were stated.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from evals.questions import Expected, Question

PASS_FACT_SHARE = 0.5


class Judgement(BaseModel):
    facts: list[bool] = Field(
        description="One entry per numbered key fact: true if the answer states it."
    )
    reason: str = Field(description="One short sentence explaining the judgement.")


JUDGE_SYSTEM = """\
You grade answers about the Model Context Protocol (MCP) specification. For each \
numbered key fact, decide whether the answer states it. Paraphrase and different \
wording count; a fact that is contradicted, missing or only hinted at does not. Do not \
use your own knowledge of MCP: judge only whether the answer states each fact."""

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", JUDGE_SYSTEM),
        ("human", "Question: {question}\n\nAnswer:\n{answer}\n\nKey facts:\n{facts}"),
    ]
)

Judge = Callable[[Question, str], tuple[list[bool], str, int]]


def make_judge(runnable: Runnable[dict[str, str], dict[str, Any]]) -> Judge:
    """Wrap a structured-output judge chain: returns (facts found, reason, tokens)."""

    def judge(question: Question, answer: str) -> tuple[list[bool], str, int]:
        facts = "\n".join(f"{n}. {fact}" for n, fact in enumerate(question.key_facts, start=1))
        reply = runnable.invoke({"question": question.question, "answer": answer, "facts": facts})
        raw = reply.get("raw")
        tokens = (raw.usage_metadata or {}).get("total_tokens", 0) if raw is not None else 0
        parsed: Judgement | None = reply["parsed"]
        if parsed is None:
            return [False] * len(question.key_facts), "judge reply unparseable", tokens
        found = (parsed.facts + [False] * len(question.key_facts))[: len(question.key_facts)]
        return found, parsed.reason, tokens

    return judge


def outcome(final: dict[str, Any]) -> str:
    """What the graph did: answered, direct, declined_out_of_scope or declined_not_covered."""
    if final.get("route") == "out_of_scope":
        return "declined_out_of_scope"
    if final.get("route") == "direct":
        return "direct"
    trace = final.get("trace", [])
    if trace and trace[-1]["step"] == "decline":
        return "declined_not_covered"
    return "answered"


def score(question: Question, final: dict[str, Any], judge: Judge) -> dict[str, Any]:
    """Score one graph result. Only answers and not_covered answers are sent to the judge."""
    did = outcome(final)
    retrieved = [d.metadata["source_path"] for d in final.get("documents", [])]
    cited = [c["source_path"] for c in final.get("citations", [])]
    expected_sources = set(question.sources)
    result: dict[str, Any] = {
        "outcome": did,
        "retrieval_hit": None,
        "facts_found": [],
        "correctness": None,
        "citation_hit": None,
        "judge_reason": "",
        "judge_tokens": 0,
        "retrieved": retrieved,
        "cited": cited,
    }

    if did == "answered" and question.expected in (Expected.ANSWER, Expected.NOT_COVERED):
        found, reason, tokens = judge(question, final.get("answer", ""))
        result.update(facts_found=found, judge_reason=reason, judge_tokens=tokens)

    share = sum(result["facts_found"]) / len(question.key_facts) if question.key_facts else 0.0
    if question.expected == Expected.ANSWER:
        result["retrieval_hit"] = bool(expected_sources & set(retrieved))
        result["correctness"] = share if did == "answered" else 0.0
        result["citation_hit"] = bool(expected_sources & set(cited))
        result["passed"] = did == "answered" and share >= PASS_FACT_SHARE
        return result
    if question.expected == Expected.NOT_COVERED:
        result["passed"] = did.startswith("declined") or share >= PASS_FACT_SHARE
        return result
    if question.expected == Expected.DECLINE:
        result["passed"] = did.startswith("declined")
        return result
    result["passed"] = did == "direct"
    return result


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Run-level metrics over the scored (non-error) results."""
    scored = [r for r in results if r.get("error") is None]
    answers = [r for r in scored if r["expected"] == Expected.ANSWER]
    declinable = (Expected.DECLINE, Expected.NOT_COVERED)
    should_decline = [r for r in scored if r["expected"] in declinable]
    correctness = [r["correctness"] for r in answers]
    return {
        "questions": len(results),
        "errors": len(results) - len(scored),
        "pass_rate": _rate([r["passed"] for r in scored]),
        "retrieval_hit_rate": _rate([r["retrieval_hit"] for r in answers]),
        "correctness": round(sum(correctness) / len(correctness), 3) if correctness else None,
        "citation_accuracy": _rate([r["citation_hit"] for r in answers]),
        "correct_declines": _rate([r["passed"] for r in should_decline]),
        "false_declines": sum(1 for r in answers if r["outcome"].startswith("declined")),
        "direct_correct": _rate([r["passed"] for r in scored if r["expected"] == Expected.DIRECT]),
        "mean_seconds": (
            round(sum(r["seconds"] for r in scored) / len(scored), 1) if scored else None
        ),
        "total_tokens": sum(r["tokens"] + r["judge_tokens"] for r in scored),
    }
