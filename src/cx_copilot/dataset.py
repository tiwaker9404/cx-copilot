"""Load the labelled tickets, and optionally sync them to a LangSmith dataset.

Each row:
    {"id": "t01", "ticket": "<latest customer message>",
     "history": [{"role": "customer"|"agent", "text": ...}, ...],      # optional, multi-turn
     "gold": {"category": ..., "escalate": bool, "required_facts": [...],
              "needs_empathy": bool, "relevant_policies": [policy ids]}}
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import Any

DATASET_NAME = "cx-copilot-tickets"
GOLD_KEYS = {"category", "escalate", "required_facts", "needs_empathy", "relevant_policies"}


def validate(rows: list[dict[str, Any]]) -> None:
    from .kb import POLICIES
    from .models import CATEGORIES

    policy_ids = {p.id for p in POLICIES}
    seen = set()
    for row in rows:
        rid = row.get("id")
        if rid in seen:
            raise ValueError(f"duplicate ticket id {rid!r}")
        seen.add(rid)
        gold = row.get("gold") or {}
        if not row.get("ticket") or set(gold) != GOLD_KEYS:
            raise ValueError(f"{rid}: needs a ticket and gold keys {sorted(GOLD_KEYS)}")
        if gold["category"] not in CATEGORIES:
            raise ValueError(f"{rid}: unknown category {gold['category']!r}")
        unknown = set(gold["relevant_policies"]) - policy_ids
        if unknown:
            raise ValueError(f"{rid}: unknown policies {sorted(unknown)}")
        for m in row.get("history") or []:
            if m.get("role") not in ("customer", "agent") or not m.get("text"):
                raise ValueError(f"{rid}: history items need role customer|agent and text")


def load_tickets() -> list[dict[str, Any]]:
    path = resources.files("cx_copilot") / "data" / "tickets.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    validate(rows)
    return rows


def content_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:8]


def sync_langsmith_dataset(name: str = DATASET_NAME) -> str:
    """Create the LangSmith dataset for the CURRENT tickets. The name carries a content hash,
    so editing tickets.jsonl creates a new dataset instead of silently comparing experiments
    run on different data. Needs LANGSMITH_API_KEY."""
    from langsmith import Client

    rows = load_tickets()
    full = f"{name}-{content_hash(rows)}"
    client = Client()
    if not client.has_dataset(dataset_name=full):
        dataset = client.create_dataset(
            full, description=f"{len(rows)} labelled support tickets for cx-copilot evals.")
        client.create_examples(
            dataset_id=dataset.id,
            examples=[{"inputs": {"ticket": r["ticket"], "history": r.get("history") or []},
                       "outputs": r["gold"], "metadata": {"ticket_id": r["id"]}}
                      for r in rows],
        )
    return full


ensure_langsmith_dataset = sync_langsmith_dataset  # old name
