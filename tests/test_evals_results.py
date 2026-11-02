import json
from pathlib import Path

import pytest

from evals import results as results_module
from evals.results import compare, describe_config, format_stamp, load_run, load_runs, main


def write_run(directory: Path, name: str, **fields: object) -> None:
    (directory / name).write_text(json.dumps(fields))


SUMMARY = {
    "pass_rate": 0.891,
    "retrieval_hit_rate": 0.889,
    "correctness": 0.819,
    "citation_accuracy": 0.889,
    "correct_declines": None,
    "false_declines": 3,
    "mean_seconds": 7.1,
    "total_tokens": 157567,
}


def test_load_runs_oldest_first_with_file_names(tmp_path: Path) -> None:
    write_run(tmp_path, "20261102T000000Z-b.json", label="b")
    write_run(tmp_path, "20261101T000000Z-a.json", label="a")
    (tmp_path / ".gitkeep").write_text("")

    runs = load_runs(tmp_path)

    assert [(r["label"], r["file"]) for r in runs] == [
        ("a", "20261101T000000Z-a.json"),
        ("b", "20261102T000000Z-b.json"),
    ]


@pytest.mark.parametrize(
    ("config", "text"),
    [
        ({"mode": "hybrid", "rerank": True, "grading": True}, "hybrid + rerank + grading"),
        ({"mode": "vector", "rerank": False, "grading": False}, "vector"),
        ({}, "?"),
    ],
)
def test_describe_config(config: dict[str, object], text: str) -> None:
    assert describe_config(config) == text


def test_compare_formats_one_row_per_run() -> None:
    runs = [
        {
            "file": "x.json",
            "label": "baseline",
            "complete": True,
            "config": {"mode": "hybrid", "rerank": True, "grading": True},
            "summary": SUMMARY,
        },
        {"file": "y.json", "complete": False, "config": {"mode": "vector"}},
    ]

    lines = compare(runs).splitlines()

    assert lines[0].split() == [
        "run", "config", "pass", "retrieval", "facts", "citations", "ok-declines",
        "false", "dec.", "time", "tokens",
    ]  # fmt: skip
    assert lines[1].split() == [
        "baseline", "hybrid", "+", "rerank", "+", "grading",
        "89%", "89%", "82%", "89%", "-", "3", "7.1s", "157567",
    ]  # fmt: skip
    assert lines[2].startswith("y.json (partial)")


def test_main_prints_the_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_run(tmp_path, "20261101T000000Z-a.json", label="a", complete=True, summary=SUMMARY)
    monkeypatch.setattr(results_module, "load_runs", lambda: load_runs(tmp_path))

    assert main() == 0
    assert "\na " in capsys.readouterr().out


def test_load_run_by_stem_only(tmp_path: Path) -> None:
    write_run(tmp_path, "20261101T000000Z-a.json", label="a")

    assert load_run("20261101T000000Z-a", tmp_path)["file"] == "20261101T000000Z-a.json"
    assert load_run("20261101T000000Z-a.json", tmp_path) is None
    assert load_run("../secrets", tmp_path) is None


@pytest.mark.parametrize(
    ("stamp", "text"),
    [("20261101T203045Z", "2026-11-01 20:30 UTC"), ("yesterday", "yesterday"), ("", "")],
)
def test_format_stamp(stamp: str, text: str) -> None:
    assert format_stamp(stamp) == text
