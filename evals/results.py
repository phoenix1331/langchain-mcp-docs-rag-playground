"""Saved eval runs: loading them and comparing their summaries."""

import json
from pathlib import Path
from typing import Any

from evals.run import RESULTS_DIR

COLUMNS = (
    ("pass_rate", "pass"),
    ("retrieval_hit_rate", "retrieval"),
    ("correctness", "correct"),
    ("citation_accuracy", "citations"),
    ("correct_declines", "declines"),
)


def load_runs(directory: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    """Every saved run, oldest first (file names start with a UTC timestamp)."""
    runs = []
    for path in sorted(directory.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        run["file"] = path.name
        runs.append(run)
    return runs


def describe_config(config: dict[str, Any]) -> str:
    """e.g. `hybrid + rerank + grading`."""
    parts = [config.get("mode", "?")]
    if config.get("rerank"):
        parts.append("rerank")
    if config.get("grading"):
        parts.append("grading")
    return " + ".join(parts)


def compare(runs: list[dict[str, Any]]) -> str:
    """A plain-text table of run summaries, one row per run."""
    header = ["run", "config", *(name for _, name in COLUMNS), "false dec.", "time", "tokens"]
    rows = [header]
    for run in runs:
        summary = run.get("summary", {})
        cells = [
            run.get("label", run["file"]) + ("" if run.get("complete") else " (partial)"),
            describe_config(run.get("config", {})),
        ]
        for key, _ in COLUMNS:
            value = summary.get(key)
            cells.append("-" if value is None else f"{value:.0%}")
        cells += [
            str(summary.get("false_declines", "-")),
            f"{summary.get('mean_seconds', '-')}s",
            str(summary.get("total_tokens", "-")),
        ]
        rows.append(cells)
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = ("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows)
    return "\n".join(line.rstrip() for line in lines)


def main() -> int:
    print(compare(load_runs()))
    return 0
