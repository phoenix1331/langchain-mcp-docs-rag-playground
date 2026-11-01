"""Publish an eval run to LangSmith as a dataset plus an experiment.

The run itself happens locally (see `evals/run.py`); this replays its stored results
through `langsmith.evaluate`, so no model is called twice. Skipped when
`LANGSMITH_API_KEY` is not set.
"""

import os
from collections.abc import Callable
from typing import Any

from langsmith import Client, evaluate

from evals.questions import Question

DATASET = "mcp-docs-questions"
METRICS = ("passed", "retrieval_hit", "correctness", "citation_hit")


def ensure_dataset(client: Client, questions: list[Question]) -> None:
    """Create the dataset if needed and add any questions it does not hold yet."""
    if not client.has_dataset(dataset_name=DATASET):
        client.create_dataset(DATASET, description="MCP docs assistant eval questions")
    existing = {ex.inputs["id"] for ex in client.list_examples(dataset_name=DATASET)}
    new = [
        {
            "inputs": {"id": q.id, "question": q.question, "spec_version": q.spec_version},
            "outputs": {"expected": q.expected, "key_facts": q.key_facts, "sources": q.sources},
            "metadata": {"category": q.category},
        }
        for q in questions
        if q.id not in existing
    ]
    if new:
        client.create_examples(dataset_name=DATASET, examples=new)


def metric(key: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def evaluator(outputs: dict[str, Any]) -> dict[str, Any]:
        value = outputs.get(key)
        return {"key": key, "score": None if value is None else float(value)}

    evaluator.__name__ = key
    return evaluator


def publish(
    payload: dict[str, Any],
    questions: list[Question],
    client: Client | None = None,
    evaluate_fn: Callable[..., Any] = evaluate,
) -> str | None:
    """Returns the experiment name, or None when LangSmith is not configured."""
    if not os.environ.get("LANGSMITH_API_KEY"):
        return None
    client = client or Client()
    ensure_dataset(client, questions)
    by_id = {r["id"]: r for r in payload["results"]}
    examples = [ex for ex in client.list_examples(dataset_name=DATASET) if ex.inputs["id"] in by_id]

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        result = by_id[inputs["id"]]
        return {"answer": result.get("answer", ""), **{k: result.get(k) for k in METRICS}}

    experiment = evaluate_fn(
        target,
        data=examples,
        evaluators=[metric(key) for key in METRICS],
        experiment_prefix=payload["label"],
        metadata=payload["config"],
        client=client,
    )
    return experiment.experiment_name
