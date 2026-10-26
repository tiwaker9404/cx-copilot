"""Run the agent over the labelled tickets and score it."""

from __future__ import annotations

from collections import defaultdict
import time
from dataclasses import dataclass, field
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
    triage_prompt: str = prompts.TRIAGE_PROMPT
    escalation: str = prompts.DEFAULT_ESCALATION
    # Not part of a run's identity, so they stay out of reports and baselines.
    cache_dir: str | None = field(default=None, compare=False)
    repeat: int = field(default=0, compare=False)
    client: Any = field(default=None, compare=False, repr=False)

    def identity(self) -> dict[str, str]:
        out = {"model": self.model, "prompt": self.prompt}
        if (self.triage_prompt, self.escalation) != (prompts.TRIAGE_PROMPT, prompts.DEFAULT_ESCALATION):
            out.update(triage_prompt=self.triage_prompt, escalation=self.escalation)
        return out


def make_target(cfg: RunConfig) -> Callable[[dict[str, Any]], dict[str, Any]]:
    model = get_model(cfg.model, cache_dir=cfg.cache_dir, repeat=cfg.repeat, client=cfg.client)
    graph = build_graph(model, cfg.prompt, triage_prompt=cfg.triage_prompt,
                        escalation=cfg.escalation)

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

    errors = 0
    started = time.perf_counter()
    for row in tickets:
        error = ""
        try:
            outputs = target({"ticket": row["ticket"]})
        except Exception as exc:  # one bad API call must not sink the whole run
            errors += 1
            error = f"{type(exc).__name__}: {exc}"[:300]
            outputs = {"reply": "", "triage": {"category": "error"}, "escalated": False,
                       "attempts": 0, "policy_ids": [], "handoff": ""}
        scores: dict[str, float | None] = {}
        for scorer in ALL_SCORERS:
            res = scorer(outputs, row["gold"])
            scores[res["key"]] = res["score"]
            if res["score"] is not None:
                by_metric[res["key"]].append(res["score"])
        triage = outputs["triage"]
        per_ticket[row["id"]] = {"scores": scores, "reply": outputs["reply"],
                                 "category": triage["category"],
                                 "urgency": triage.get("urgency"),
                                 "sentiment": triage.get("sentiment"),
                                 "escalated": outputs["escalated"], "error": error,
                                 "escalated_by": ("" if not outputs["escalated"] else
                                                  "qa" if outputs["handoff"].startswith("QA failed")
                                                  else "triage"),
                                 "ticket": row["ticket"], "gold": row["gold"]}

    return {
        "config": cfg.identity(),
        "n": len(tickets),
        "errors": errors,
        "seconds": round(time.perf_counter() - started, 2),
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
        metadata=cfg.identity(),
        max_concurrency=4,
    )
    return results.experiment_name
