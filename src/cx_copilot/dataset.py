"""Load the labelled tickets, and optionally sync them to a LangSmith dataset."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

DATASET_NAME = "cx-copilot-tickets"


def load_tickets() -> list[dict[str, Any]]:
    path = resources.files("cx_copilot") / "data" / "tickets.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ticket ids in tickets.jsonl")
    return rows


def ensure_langsmith_dataset(name: str = DATASET_NAME) -> str:
    """Create the dataset in LangSmith if it does not exist yet. Needs LANGSMITH_API_KEY."""
    from langsmith import Client

    client = Client()
    if not client.has_dataset(dataset_name=name):
        dataset = client.create_dataset(
            name, description="Labelled support tickets for cx-copilot regression evals."
        )
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {"inputs": {"ticket": r["ticket"]}, "outputs": r["gold"],
                 "metadata": {"ticket_id": r["id"]}}
                for r in load_tickets()
            ],
        )
    return name
