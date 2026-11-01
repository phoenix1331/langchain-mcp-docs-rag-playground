"""Plain-language descriptions of each graph stage, for live progress in the CLI and UI."""

from typing import Any

from app.graph.nodes import relevant_documents


def describe_start(step: str, state: dict[str, Any]) -> str:
    """What a node is about to do, from the state it was given."""
    if step == "route":
        return "Deciding whether this needs the docs..."
    if step == "retrieve":
        return f"Searching the {state['spec_version']} docs for {state['query']!r}..."
    if step == "grade":
        return f"Checking which of the {len(state.get('documents', []))} chunks are relevant..."
    if step == "rewrite":
        return "Too few relevant chunks, so rephrasing the search..."
    if step == "answer" and state.get("route") == "direct":
        return "Replying..."
    if step == "answer":
        return f"Writing the answer from {len(relevant_documents(state))} relevant chunks..."
    if step == "decline":
        return "Explaining why this can't be answered..."
    return f"Running {step}..."


def describe_result(update: dict[str, Any]) -> str:
    """One line for a finished node: time, tokens and its trace summary."""
    steps = update.get("trace") or []
    if not steps:
        return "  done"
    step = steps[-1]
    tokens = f", {step['tokens']} tokens" if step["tokens"] else ""
    summary = f": {step['summary']}" if step["summary"] else ""
    return f"  done in {step['ms'] / 1000:.1f}s{tokens}{summary}"
