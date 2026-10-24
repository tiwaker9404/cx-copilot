"""Regression gate: block a prompt or model change that makes support worse.

    cx-gate                               # check the default config against evals/baseline.json
    cx-gate --prompt reply_v3_friendly    # check a candidate prompt
    cx-gate --model fake:lite             # check a candidate model
    cx-gate --update-baseline             # accept the candidate as the new baseline (only if it passes)

Exit codes: 0 = pass, 1 = regression detected, 2 = usage or setup error.

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

from . import prompts
from .runner import RunConfig, run_langsmith_experiment, run_suite

DEFAULT_BASELINE = Path("evals/baseline.json")

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
}
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
        "n": report["n"],
        "metrics": report["metrics"],
        "ticket_quality": {tid: d["scores"]["quality"] for tid, d in report["tickets"].items()
                           if d["scores"].get("quality") is not None},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cx-gate", description=__doc__.split("\n\n")[0])
    p.add_argument("--prompt", default=prompts.DEFAULT_REPLY_PROMPT,
                   help=f"reply prompt name (one of: {', '.join(prompts.available())})")
    p.add_argument("--model", default=os.environ.get("CX_MODEL", "fake"),
                   help="fake | fake:lite | anthropic:<model> | openai:<model>")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--update-baseline", action="store_true",
                   help="write the candidate's scores as the new baseline (only if the gate passes)")
    p.add_argument("--trace", action="store_true",
                   help="send traces to LangSmith (needs LANGSMITH_API_KEY)")
    p.add_argument("--experiment", action="store_true",
                   help="also run it as a LangSmith experiment for side-by-side comparison")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = p.parse_args(argv)

    cfg = RunConfig(model=args.model, prompt=args.prompt)
    try:
        prompts.load(cfg.prompt)
        with tracing_context(enabled=bool(args.trace)):
            report = run_suite(cfg)
        if args.experiment:
            print(f"LangSmith experiment: {run_langsmith_experiment(cfg)}", file=sys.stderr)
    except Exception as exc:  # setup problems are exit 2, not a regression
        print(f"cx-gate: error: {exc}", file=sys.stderr)
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
    result = compare(baseline, report)
    markdown = render_markdown(result, cfg, baseline["config"])

    if args.json:
        print(json.dumps({**result, "candidate": report["metrics"]}, indent=2))
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
