from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from evals.langsmith_sync import DATASET, METRICS, ensure_dataset, metric, publish
from evals.questions import load_questions

QUESTIONS = load_questions()[:3]


def example(qid: str) -> SimpleNamespace:
    return SimpleNamespace(inputs={"id": qid})


def test_publish_is_skipped_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    assert publish({"results": []}, QUESTIONS, client=MagicMock()) is None


def test_ensure_dataset_creates_it_and_adds_only_new_examples() -> None:
    client = MagicMock()
    client.has_dataset.return_value = False
    client.list_examples.return_value = [example(QUESTIONS[0].id)]

    ensure_dataset(client, QUESTIONS)

    client.create_dataset.assert_called_once()
    added = client.create_examples.call_args.kwargs["examples"]
    assert [e["inputs"]["id"] for e in added] == [q.id for q in QUESTIONS[1:]]
    assert added[0]["outputs"]["key_facts"] == QUESTIONS[1].key_facts
    assert added[0]["metadata"] == {"category": QUESTIONS[1].category}


def test_ensure_dataset_with_nothing_new() -> None:
    client = MagicMock()
    client.has_dataset.return_value = True
    client.list_examples.return_value = [example(q.id) for q in QUESTIONS]

    ensure_dataset(client, QUESTIONS)

    client.create_dataset.assert_not_called()
    client.create_examples.assert_not_called()


def test_metric_scores_and_none() -> None:
    assert metric("passed")({"passed": True}) == {"key": "passed", "score": 1.0}
    assert metric("correctness")({"correctness": None}) == {"key": "correctness", "score": None}
    assert metric("citation_hit").__name__ == "citation_hit"


def test_publish_replays_stored_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "key")
    client = MagicMock()
    client.has_dataset.return_value = True
    client.list_examples.return_value = [example(q.id) for q in QUESTIONS]
    payload = {
        "label": "baseline",
        "config": {"mode": "hybrid"},
        "results": [{"id": QUESTIONS[0].id, "answer": "a", "passed": True, "correctness": 1.0}],
    }
    calls: list[dict[str, Any]] = []

    def fake_evaluate(target: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        outputs = target({"id": QUESTIONS[0].id})
        assert outputs["answer"] == "a"
        assert set(outputs) == {"answer", *METRICS}
        return SimpleNamespace(experiment_name="baseline-123")

    assert publish(payload, QUESTIONS, client=client, evaluate_fn=fake_evaluate) == "baseline-123"
    [kwargs] = calls
    assert [e.inputs["id"] for e in kwargs["data"]] == [QUESTIONS[0].id]
    assert kwargs["experiment_prefix"] == "baseline"
    assert kwargs["metadata"] == {"mode": "hybrid"}
    assert [e.__name__ for e in kwargs["evaluators"]] == list(METRICS)
    assert DATASET == "mcp-docs-questions"
