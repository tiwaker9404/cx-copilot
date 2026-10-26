"""Run the agent over the labelled tickets and score it."""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable

from . import kb, prompts
from .dataset import load_tickets, sync_langsmith_dataset
from .graph import build_graph, run_ticket
from .judge import get_judge
from .models import get_model
from .rules import issue_kind
from .scorers import ALL_SCORERS


@dataclass(frozen=True)
class RunConfig:
    model: str = "fake"
    prompt: str = prompts.DEFAULT_REPLY_PROMPT
    triage_prompt: str = prompts.TRIAGE_PROMPT
    escalation: str = prompts.DEFAULT_ESCALATION
    retrieval: str = kb.DEFAULT_METHOD
    embed: str | None = None          # embedding model for retrieval "embed"/"hybrid"
    judge: str | None = None          # judge model; adds judge_* metrics
    # Not part of a run's identity, so they stay out of reports and baselines.
    cache_dir: str | None = field(default=None, compare=False)
    repeat: int = field(default=0, compare=False)
    client: Any = field(default=None, compare=False, repr=False)
    progress: bool = field(default=False, compare=False)

    def identity(self) -> dict[str, str]:
        out = {"model": self.model, "prompt": self.prompt}
        if (self.triage_prompt, self.escalation) != (prompts.TRIAGE_PROMPT, prompts.DEFAULT_ESCALATION):
            out.update(triage_prompt=self.triage_prompt, escalation=self.escalation)
        if self.retrieval != kb.DEFAULT_METHOD or self.embed:
            out["retrieval"] = self.retrieval + (f"+{self.embed}" if self.embed else "")
        if self.judge:
            out["judge"] = self.judge
        return out


def make_target(cfg: RunConfig) -> Callable[[dict[str, Any]], dict[str, Any]]:
    model = get_model(cfg.model, cache_dir=cfg.cache_dir, repeat=cfg.repeat, client=cfg.client)
    embedder = kb.get_embedder(cfg.embed, cache_dir=cfg.cache_dir)
    retriever = kb.Retriever(method=cfg.retrieval, embedder=embedder)
    graph = build_graph(model, cfg.prompt, triage_prompt=cfg.triage_prompt,
                        escalation=cfg.escalation, retriever=retriever)
    judge = get_judge(cfg.judge, cache_dir=cfg.cache_dir)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        history = inputs.get("history") or []
        state = run_ticket(graph, inputs["ticket"], history=history, model_name=model.name,
                           reply_prompt=cfg.prompt, triage_prompt=cfg.triage_prompt,
                           escalation=cfg.escalation, retrieval=retriever.name)
        out = {
            "ticket": inputs["ticket"],
            "history": history,
            "reply": state["reply"],
            "triage": state["triage"],
            "escalated": state.get("escalated", False),
            "escalation_reason": state.get("escalation_reason", ""),
            "attempts": state.get("attempts", 0),
            "policy_ids": state.get("policy_ids", []),
            "policies": state.get("policies", []),
            "first_issues": state.get("first_issues", []),
            "first_draft": state.get("first_draft", "") if state.get("first_issues") else "",
            "handoff": state.get("handoff", ""),
            "usage": state.get("usage", {}),
        }
        if judge is not None and not out["escalated"]:
            out["judge"] = judge(out["ticket"], history, out["policies"], out["reply"])
        return out

    return target


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(q / 100 * len(ordered))))
    return ordered[rank - 1]


def _perf(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usage = [r["usage"] for r in rows if r.get("usage")]
    model_ms = [u["model_ms"] for u in usage if u.get("latency_known")]
    tokens = [u["input_tokens"] + u["output_tokens"] for u in usage]
    wall = [r["wall_ms"] for r in rows]
    return {
        "tickets": len(rows),
        "mean_calls_per_ticket": round(fmean(u["calls"] for u in usage), 3) if usage else None,
        "mean_tokens_per_ticket": round(fmean(tokens), 1) if tokens else None,
        "max_tokens_per_ticket": max(tokens) if tokens else None,
        "p50_model_ms_per_ticket": percentile(model_ms, 50),
        "p95_model_ms_per_ticket": percentile(model_ms, 95),
        "latency_known_for": len(model_ms),
        "p50_wall_ms_per_ticket": percentile(wall, 50),
        "p95_wall_ms_per_ticket": percentile(wall, 95),
    }


def run_suite(cfg: RunConfig, tickets: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run every ticket through the graph, score it, and aggregate. Fully offline with `fake`."""
    tickets = tickets if tickets is not None else load_tickets()
    target = make_target(cfg)
    per_ticket: dict[str, Any] = {}
    by_metric: dict[str, list[float]] = defaultdict(list)
    perf_rows: list[dict[str, Any]] = []
    caught: dict[str, int] = defaultdict(int)

    errors = 0
    started = time.perf_counter()
    for n, row in enumerate(tickets, 1):
        error = ""
        t0 = time.perf_counter()
        inputs = {"ticket": row["ticket"], "history": row.get("history") or []}
        try:
            outputs = target(inputs)
        except Exception as exc:  # one bad API call must not sink the whole run
            errors += 1
            error = f"{type(exc).__name__}: {exc}"[:300]
            outputs = {**inputs, "reply": "", "triage": {"category": "error"},
                       "escalated": False, "escalation_reason": "", "attempts": 0,
                       "policy_ids": [], "policies": [], "first_issues": [], "handoff": "",
                       "usage": {}}
        wall_ms = int((time.perf_counter() - t0) * 1000)
        if not error:
            perf_rows.append({"usage": outputs.get("usage"), "wall_ms": wall_ms})
        scores: dict[str, float | None] = {}
        for scorer in ALL_SCORERS:
            res = scorer(outputs, row["gold"])
            scores[res["key"]] = res["score"]
            if res["score"] is not None:
                by_metric[res["key"]].append(res["score"])
        for issue in outputs.get("first_issues") or []:
            caught[issue_kind(issue)] += 1
        triage = outputs["triage"]
        per_ticket[row["id"]] = {"scores": scores, "reply": outputs["reply"],
                                 "category": triage["category"],
                                 "urgency": triage.get("urgency"),
                                 "sentiment": triage.get("sentiment"),
                                 "escalated": outputs["escalated"], "error": error,
                                 "escalated_by": outputs.get("escalation_reason", "")
                                 if outputs["escalated"] else "",
                                 "policy_ids": outputs.get("policy_ids", []),
                                 "first_issues": outputs.get("first_issues", []),
                                 "first_draft": outputs.get("first_draft", ""),
                                 "judge": outputs.get("judge"),
                                 "usage": outputs.get("usage", {}), "wall_ms": wall_ms,
                                 "turns": 1 + sum(m.get("role") == "customer"
                                                  for m in inputs["history"]),
                                 "ticket": row["ticket"], "gold": row["gold"]}
        if cfg.progress:
            print(f"  [{n}/{len(tickets)}] {row['id']:<5} {wall_ms / 1000:5.1f}s"
                  f"{'  ERROR ' + error[:60] if error else ''}", file=sys.stderr, flush=True)

    return {
        "config": cfg.identity(),
        "n": len(tickets),
        "errors": errors,
        "seconds": round(time.perf_counter() - started, 2),
        "metrics": {k: round(fmean(v), 4) for k, v in sorted(by_metric.items())},
        "counts": {k: len(v) for k, v in sorted(by_metric.items())},
        "perf": _perf(perf_rows),
        "qa_caught_first_draft": dict(sorted(caught.items())),
        "tickets": per_ticket,
    }


def run_langsmith_experiment(cfg: RunConfig, experiment_prefix: str | None = None) -> str:
    """Run the same agent and scorers as a LangSmith experiment. Needs LANGSMITH_API_KEY.

    The gate does not use this path to decide pass/fail; it is for browsing results,
    comparing experiments side by side and drilling into traces in the LangSmith UI.
    """
    from langsmith import evaluate

    results = evaluate(
        make_target(cfg),
        data=sync_langsmith_dataset(),
        evaluators=list(ALL_SCORERS),
        experiment_prefix=experiment_prefix or f"{cfg.prompt}__{cfg.model}".replace(":", "-"),
        metadata=cfg.identity(),
        max_concurrency=1 if cfg.model.startswith("ollama") else 4,
    )
    return results.experiment_name
