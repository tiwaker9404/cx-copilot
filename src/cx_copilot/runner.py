"""Run the agent over the labelled tickets and score it."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Callable

from . import prompts
from .dataset import DATASET_NAME, ensure_langsmith_dataset, load_tickets
from .graph import build_graph, run_ticket
from .models import get_model
from .scorers import ALL_SCORERS


@dataclass(frozen=True)
class RunConfig:
    model: str = "fake"
    prompt: str = prompts.DEFAULT_REPLY_PROMPT


def make_target(cfg: RunConfig) -> Callable[[dict[str, Any]], dict[str, Any]]:
    model = get_model(cfg.model)
    graph = build_graph(model, cfg.prompt)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        state = run_ticket(graph, inputs["ticket"], model_name=model.name,
                           reply_prompt=cfg.prompt)
        return {
            "reply": state["reply"],
            "triage": state["triage"],
            "escalated": state.get("escalated", False),
            "attempts": state.get("attempts", 0),
            "policy_ids": state.get("policy_ids", []),
            "handoff": state.get("handoff", ""),
        }

    return target


def run_suite(cfg: RunConfig, tickets: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run every ticket through the graph, score it, and aggregate. Fully offline with `fake`."""
    tickets = tickets if tickets is not None else load_tickets()
    target = make_target(cfg)
    per_ticket: dict[str, Any] = {}
    by_metric: dict[str, list[float]] = defaultdict(list)

    for row in tickets:
        outputs = target({"ticket": row["ticket"]})
        scores: dict[str, float | None] = {}
        for scorer in ALL_SCORERS:
            res = scorer(outputs, row["gold"])
            scores[res["key"]] = res["score"]
            if res["score"] is not None:
                by_metric[res["key"]].append(res["score"])
        per_ticket[row["id"]] = {"scores": scores, "reply": outputs["reply"],
                                 "category": outputs["triage"]["category"],
                                 "escalated": outputs["escalated"]}

    return {
        "config": asdict(cfg),
        "n": len(tickets),
        "metrics": {k: round(fmean(v), 4) for k, v in sorted(by_metric.items())},
        "counts": {k: len(v) for k, v in sorted(by_metric.items())},
        "tickets": per_ticket,
    }


def run_langsmith_experiment(cfg: RunConfig) -> str:
    """Run the same agent and scorers as a LangSmith experiment. Needs LANGSMITH_API_KEY.

    The gate does not use this path to decide pass/fail; it is for browsing results,
    comparing experiments side by side and drilling into traces in the LangSmith UI.
    """
    from langsmith import evaluate

    results = evaluate(
        make_target(cfg),
        data=ensure_langsmith_dataset(DATASET_NAME),
        evaluators=list(ALL_SCORERS),
        experiment_prefix=f"{cfg.prompt}__{cfg.model}".replace(":", "-"),
        metadata=asdict(cfg),
        max_concurrency=4,
    )
    return results.experiment_name
