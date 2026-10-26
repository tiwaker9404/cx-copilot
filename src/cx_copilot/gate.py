"""Regression gate: block a prompt or model change that makes support worse.

    cx-gate                               # check the default config against evals/baseline.json
    cx-gate --prompt reply_v3_friendly    # check a candidate prompt
    cx-gate --model fake:lite             # check a candidate model
    cx-gate --update-baseline             # accept the candidate as the new baseline (only if it passes)
    cx-gate --model ollama:qwen2.5:7b --cache .cx_cache --budgets evals/budgets.json

Exit codes: 0 = pass, 1 = regression or budget exceeded, 2 = usage or setup error.

Budgets (evals/budgets.json) cap cost and latency per ticket: mean model calls, mean tokens,
and p95 model time. A prompt that adds a retry on every ticket can keep quality and still
double the bill; the budget check catches that.

Why per-metric limits instead of one average? A prompt can lift the average while one
dimension quietly collapses (say, it stops quoting policy). Each metric gets its own
tolerance, and safety and escalation recall get zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from langsmith import tracing_context

from . import kb, prompts
from .dataset import content_hash, load_tickets
from .models import missing_key
from .runner import RunConfig, run_langsmith_experiment, run_suite

DEFAULT_BASELINE = Path("evals/baseline.json")
DEFAULT_BUDGETS = Path("evals/budgets.json")
BUDGET_KEYS = ("mean_calls_per_ticket", "mean_tokens_per_ticket", "p95_model_ms_per_ticket")

# metric -> largest allowed drop versus baseline
MAX_DROP: dict[str, float] = {
    "quality": 0.02,
    "category_accuracy": 0.05,
    "policy_grounded": 0.05,
    "empathy": 0.05,
    "actionable": 0.05,
    "length_ok": 0.10,
    "first_try_clean": 0.10,
    "escalation_correct": 0.05,
    "escalation_recall": 0.0,  # missing an escalation is the costly error
    "safety": 0.0,
    "no_invented_commitments": 0.0,  # a promise the business never made, sent to a customer
    "no_repeat_questions": 0.05,
    "draft_commitment_free": 0.05,
    "followup_not_repeated": 0.10,
    "retrieval_recall": 0.05,
}
# Reported, never gated: on real runs the judge was only trustworthy for some criteria (see
# evals/judge_calibration.md), and a gate must not depend on an uncalibrated model.
JUDGE_METRICS = ("judge_grounded", "judge_answers_question", "judge_tone_ok",
                 "judge_no_invented_commitments")
EPS = 1e-9


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for metric, limit in MAX_DROP.items():
        base = baseline["metrics"].get(metric)
        cand = candidate["metrics"].get(metric)
        if base is None or cand is None:
            rows.append({"metric": metric, "baseline": base, "candidate": cand, "delta": None,
                         "limit": limit, "status": "SKIP"})
            continue
        delta = round(cand - base, 4)
        rows.append({"metric": metric, "baseline": base, "candidate": cand, "delta": delta,
                     "limit": limit, "status": "FAIL" if delta < -limit - EPS else "ok"})

    regressed = []
    base_q = baseline.get("ticket_quality", {})
    for tid, data in candidate["tickets"].items():
        cq, bq = data["scores"].get("quality"), base_q.get(tid)
        if cq is not None and bq is not None and cq < bq - EPS:
            regressed.append({"id": tid, "baseline": bq, "candidate": round(cq, 4)})
    regressed.sort(key=lambda r: r["candidate"] - r["baseline"])

    return {"passed": not any(r["status"] == "FAIL" for r in rows), "rows": rows,
            "regressed_tickets": regressed}


def budget_for(budgets: dict[str, Any], model: str) -> dict[str, Any]:
    """The budget for this model spec, falling back to "default"."""
    return {**budgets.get("default", {}), **budgets.get(model, {})}


def check_budgets(perf: dict[str, Any], budget: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in BUDGET_KEYS:
        limit, actual = budget.get(key), perf.get(key)
        if limit is None or actual is None:
            status = "SKIP"
        else:
            status = "FAIL" if actual > limit + EPS else "ok"
        rows.append({"budget": key, "limit": limit, "actual": actual, "status": status})
    return rows


def render_budgets(rows: list[dict[str, Any]], perf: dict[str, Any]) -> str:
    fmt = lambda v: "n/a" if v is None else (f"{v:,.0f}" if v >= 100 else f"{v:.2f}")  # noqa: E731
    lines = ["", "| Budget (per ticket) | Limit | Actual | Status |", "|---|---|---|---|"]
    lines += [f"| {r['budget']} | {fmt(r['limit'])} | {fmt(r['actual'])} | {r['status']} |"
              for r in rows]
    if perf.get("latency_known_for", 0) < perf.get("tickets", 0):
        lines.append(f"\nLatency was recorded for {perf.get('latency_known_for', 0)} of "
                     f"{perf.get('tickets', 0)} tickets (older cache entries have none).")
    return "\n".join(lines)


def render_markdown(result: dict[str, Any], cfg: RunConfig, baseline_cfg: dict[str, Any]) -> str:
    verdict = "PASS" if result["passed"] else "BLOCKED"
    lines = [
        f"## cx-copilot regression gate: {verdict}",
        "",
        f"Candidate: `{cfg.prompt}` on `{cfg.model}`  |  Baseline: "
        f"`{baseline_cfg.get('prompt')}` on `{baseline_cfg.get('model')}`",
        "",
        "| Metric | Baseline | Candidate | Change | Allowed drop | Status |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        if r["baseline"] is None and r["candidate"] is None:
            continue  # e.g. judge metrics when no judge ran
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        delta = "n/a" if r["delta"] is None else f"{r['delta']:+.3f}"
        lines.append(f"| {r['metric']} | {fmt(r['baseline'])} | {fmt(r['candidate'])} | "
                     f"{delta} | {r['limit']:.2f} | {r['status']} |")
    worst = result["regressed_tickets"][:5]
    if worst:
        lines += ["", "Biggest per-ticket drops: " + ", ".join(
            f"{w['id']} ({w['baseline']:.2f} -> {w['candidate']:.2f})" for w in worst)]
    return "\n".join(lines)


def _baseline_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": report["config"],
        "dataset": content_hash(load_tickets()),
        "n": report["n"],
        "metrics": report["metrics"],
        "perf": report.get("perf", {}),
        "ticket_quality": {tid: d["scores"]["quality"] for tid, d in report["tickets"].items()
                           if d["scores"].get("quality") is not None},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cx-gate", description=__doc__.split("\n\n")[0])
    p.add_argument("--prompt", default=prompts.DEFAULT_REPLY_PROMPT,
                   help=f"reply prompt name (one of: {', '.join(prompts.available())})")
    p.add_argument("--model", default=os.environ.get("CX_MODEL", "fake"),
                   help="fake | fake:lite | anthropic:<model> | openai:<model>")
    p.add_argument("--triage-prompt", default=os.environ.get("CX_TRIAGE_PROMPT", prompts.TRIAGE_PROMPT))
    p.add_argument("--escalation", choices=["model", "rules"],
                   default=os.environ.get("CX_ESCALATION", prompts.DEFAULT_ESCALATION))
    p.add_argument("--retrieval", choices=kb.METHODS,
                   default=os.environ.get("CX_RETRIEVAL", kb.DEFAULT_METHOD))
    p.add_argument("--embed", default=os.environ.get("CX_EMBED_MODEL") or None, metavar="SPEC",
                   help="embedding model for retrieval, e.g. ollama:nomic-embed-text")
    p.add_argument("--judge", metavar="SPEC", default=None,
                   help="also score replies with an LLM judge, e.g. ollama:qwen2.5:7b")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--budgets", type=Path, default=None,
                   help=f"check cost/latency budgets (default: {DEFAULT_BUDGETS} if it exists)")
    p.add_argument("--no-budgets", action="store_true", help="skip the budget check")
    p.add_argument("--progress", action="store_true", help="print a line per ticket")
    p.add_argument("--update-baseline", action="store_true",
                   help="write the candidate's scores as the new baseline (only if the gate passes)")
    p.add_argument("--cache", default=os.environ.get("CX_CACHE_DIR"), metavar="DIR",
                   help="cache live model responses here so reruns are free and reproducible")
    p.add_argument("--allow-errors", action="store_true",
                   help="score even if some tickets hit API errors (they count as failures)")
    p.add_argument("--trace", action="store_true",
                   help="send traces to LangSmith (needs LANGSMITH_API_KEY)")
    p.add_argument("--experiment", action="store_true",
                   help="also run it as a LangSmith experiment for side-by-side comparison")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = p.parse_args(argv)

    cfg = RunConfig(model=args.model, prompt=args.prompt, triage_prompt=args.triage_prompt,
                    escalation=args.escalation, retrieval=args.retrieval, embed=args.embed,
                    judge=args.judge, cache_dir=args.cache, progress=args.progress)
    if (var := missing_key(cfg.model)):
        print(f"cx-gate: error: {cfg.model} needs {var} to be set", file=sys.stderr)
        return 2
    try:
        prompts.load(cfg.prompt)
        prompts.load(cfg.triage_prompt)
        if args.judge and (var := missing_key(args.judge)):
            raise ValueError(f"judge {args.judge} needs {var} to be set")
        with tracing_context(enabled=bool(args.trace)):
            report = run_suite(cfg)
        if args.experiment:
            print(f"LangSmith experiment: {run_langsmith_experiment(cfg)}", file=sys.stderr)
    except Exception as exc:  # setup problems are exit 2, not a regression
        print(f"cx-gate: error: {exc}", file=sys.stderr)
        return 2

    if report["errors"] and not args.allow_errors:
        bad = [f"{tid}: {d['error']}" for tid, d in report["tickets"].items() if d["error"]]
        print(f"cx-gate: {report['errors']} of {report['n']} tickets hit errors, so the scores "
              "are not comparable. First error:\n  " + bad[0], file=sys.stderr)
        return 2

    if not args.baseline.exists():
        if args.update_baseline:
            args.baseline.parent.mkdir(parents=True, exist_ok=True)
            args.baseline.write_text(json.dumps(_baseline_payload(report), indent=2) + "\n")
            print(f"Wrote new baseline to {args.baseline}")
            return 0
        print(f"cx-gate: no baseline at {args.baseline}. Create one with --update-baseline.",
              file=sys.stderr)
        return 2

    baseline = json.loads(args.baseline.read_text())
    current = content_hash(load_tickets())
    if baseline.get("dataset") != current:
        if args.update_baseline:  # new tickets: start a new baseline, nothing to compare to
            args.baseline.write_text(json.dumps(_baseline_payload(report), indent=2) + "\n")
            print(f"The dataset changed ({baseline.get('dataset')} -> {current}), so the "
                  f"baseline was rebuilt from this run: {args.baseline}")
            return 0
        print(f"cx-gate: {args.baseline} was made on a different dataset "
              f"({baseline.get('n')} tickets, {baseline.get('dataset')}) than the current one "
              f"({report['n']} tickets, {current}); the scores are not comparable. Rebuild it "
              "from the current default with: cx-gate --update-baseline", file=sys.stderr)
        return 2
    result = compare(baseline, report)
    markdown = render_markdown(result, cfg, baseline["config"])

    budget_path = args.budgets or (DEFAULT_BUDGETS if DEFAULT_BUDGETS.exists() else None)
    budget_rows: list[dict[str, Any]] = []
    if budget_path and not args.no_budgets:
        budget_rows = check_budgets(report["perf"],
                                    budget_for(json.loads(budget_path.read_text()), cfg.model))
        markdown += "\n" + render_budgets(budget_rows, report["perf"])
        if any(r["status"] == "FAIL" for r in budget_rows):
            result["passed"] = False
            markdown = markdown.replace("gate: PASS", "gate: BLOCKED", 1)
    if report.get("qa_caught_first_draft"):
        markdown += "\n\nQA guardrail caught in first drafts: " + ", ".join(
            f"{k} x{v}" for k, v in report["qa_caught_first_draft"].items())

    if args.json:
        print(json.dumps({**result, "candidate": report["metrics"], "perf": report["perf"],
                          "budgets": budget_rows}, indent=2))
    else:
        print(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")

    if args.update_baseline:
        if result["passed"]:
            args.baseline.write_text(json.dumps(_baseline_payload(report), indent=2) + "\n")
            print(f"Baseline updated: {args.baseline}")
        else:
            print("Not updating the baseline because the gate failed.", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
