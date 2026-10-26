"""Run the same tickets through several models (or prompt variants) and see what changes.

    cx-compare --models ollama:qwen2.5:7b ollama:llama3.2 --repeats 3 --cache .cx_cache
    cx-compare --models ollama:llama3.2 ollama:llama3.2@triage_v2/rules --cache .cx_cache

Each entry is MODEL[@TRIAGE_PROMPT[/ESCALATION]]. The first entry is the reference: for every
other one the report says whether the regression gate would block the swap, which metrics moved,
and exactly which tickets went wrong (missed escalations first, because they are the costly
ones). Results go to the terminal and to `evals/compare.md` / `evals/compare.json`.

Live models are nondeterministic, so use `--repeats 3` before believing a difference. With
`--cache`, rerunning the same command makes no model calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

from langsmith import tracing_context

from . import prompts
from .gate import MAX_DROP
from .gate import compare as gate_compare
from .graph import ESCALATION_MODES
from .models import LIVE_PROVIDERS, get_model, missing_key
from .runner import RunConfig, run_suite

NOTABLE = 0.10          # call out any metric that moves by at least this much
CALLS_PER_TICKET = 3    # upper-ish bound: 1 triage + up to 2 reply drafts


def parse_spec(spec: str, default_triage: str, default_escalation: str) -> tuple[str, str, str]:
    """'ollama:qwen2.5:7b@triage_v2/rules' -> ('ollama:qwen2.5:7b', 'triage_v2', 'rules')."""
    model, _, variant = spec.partition("@")
    triage, _, escalation = variant.partition("/")
    triage, escalation = triage or default_triage, escalation or default_escalation
    if escalation not in ESCALATION_MODES:
        raise ValueError(f"{spec}: escalation must be one of {ESCALATION_MODES}")
    prompts.load(triage)  # raises with the list of available prompts
    return model, triage, escalation


def _misses(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per ticket: how many repeats missed an escalation, over-escalated, or got the category
    wrong, plus one example of what the model did."""
    out: dict[str, Any] = {}
    for tid, first in runs[0]["tickets"].items():
        gold = first["gold"]
        counts = {"missed_escalation": 0, "over_escalation": 0, "wrong_category": 0,
                  "error": 0}
        example = None
        for r in runs:
            t = r["tickets"][tid]
            bad = False
            if t["error"]:
                counts["error"] += 1
                bad = True
            if gold["escalate"] and not t["escalated"]:
                counts["missed_escalation"] += 1
                bad = True
            if not gold["escalate"] and t["escalated"]:
                counts["over_escalation"] += 1
                bad = True
            if t["category"] != gold["category"]:
                counts["wrong_category"] += 1
                bad = True
            if bad and example is None:
                example = {k: t.get(k) for k in ("category", "urgency", "sentiment",
                                                  "escalated", "escalated_by", "error")}
                example["reply"] = (t.get("reply") or "")[:240]
        if any(counts.values()):
            out[tid] = {"ticket": first["ticket"], "gold": gold, "counts": counts,
                        "repeats": len(runs), "example": example}
    return out


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Average several repeats of one entry; keep the min-to-max spread per metric."""
    metrics: dict[str, float] = {}
    spread: dict[str, float] = {}
    for key in sorted({k for r in runs for k in r["metrics"]}):
        vals = [r["metrics"][key] for r in runs if key in r["metrics"]]
        metrics[key] = round(fmean(vals), 4)
        spread[key] = round(max(vals) - min(vals), 4)
    tickets: dict[str, Any] = {}
    for tid in runs[0]["tickets"]:
        q = [r["tickets"][tid]["scores"].get("quality") for r in runs]
        q = [v for v in q if v is not None]
        tickets[tid] = {"scores": {"quality": fmean(q) if q else None}}
    has_detail = all("gold" in t for t in runs[0]["tickets"].values())
    return {"metrics": metrics, "spread": spread, "tickets": tickets,
            "misses": _misses(runs) if has_detail else {},
            "errors": sum(r["errors"] for r in runs),
            "seconds": round(sum(r["seconds"] for r in runs), 1), "n": runs[0]["n"],
            "repeats": len(runs)}


def run_compare(specs: list[str], prompt: str, repeats: int = 1, cache_dir: str | None = None,
                triage_prompt: str = prompts.TRIAGE_PROMPT,
                escalation: str = prompts.DEFAULT_ESCALATION) -> dict[str, Any]:
    results = {}
    for spec in specs:
        model, triage, esc = parse_spec(spec, triage_prompt, escalation)
        results[spec] = _aggregate([
            run_suite(RunConfig(model, prompt, triage_prompt=triage, escalation=esc,
                                cache_dir=cache_dir, repeat=i))
            for i in range(repeats)])
    ref = specs[0]
    verdicts = {}
    for spec in specs[1:]:
        g = gate_compare({"metrics": results[ref]["metrics"],
                          "ticket_quality": {t: d["scores"]["quality"]
                                             for t, d in results[ref]["tickets"].items()
                                             if d["scores"]["quality"] is not None}},
                         {"metrics": results[spec]["metrics"], "tickets": results[spec]["tickets"]})
        verdicts[spec] = {"passed": g["passed"],
                          "failed_metrics": [r["metric"] for r in g["rows"] if r["status"] == "FAIL"]}
    return {"prompt": prompt, "reference": ref, "repeats": repeats, "results": results,
            "verdicts": verdicts}


def _miss_lines(misses: dict[str, Any], kind: str, limit: int = 8) -> list[str]:
    rows = sorted(((tid, m) for tid, m in misses.items() if m["counts"][kind]),
                  key=lambda x: -x[1]["counts"][kind])
    lines = []
    for tid, m in rows[:limit]:
        ex = m["example"] or {}
        got = (f"got {ex.get('category')}/{ex.get('urgency')}/{ex.get('sentiment')}"
               + (", escalated by the QA fallback (replies failed checks twice)"
                  if ex.get("escalated_by") == "qa" else ", escalated" if ex.get("escalated") else ""))
        want = f"want {m['gold']['category']}{', escalate' if m['gold']['escalate'] else ''}"
        text = m["ticket"] if len(m["ticket"]) <= 90 else m["ticket"][:87] + "..."
        lines.append(f"  - `{tid}` ({m['counts'][kind]}/{m['repeats']}) \"{text}\" - {got}; {want}")
    if len(rows) > limit:
        lines.append(f"  - ...and {len(rows) - limit} more in the JSON")
    return lines


def render(report: dict[str, Any]) -> str:
    specs = list(report["results"])
    ref, res = report["reference"], report["results"]
    out = [f"# Comparison: reply prompt `{report['prompt']}`, {res[ref]['n']} tickets, "
           f"{report['repeats']} repeat(s)", "",
           f"Reference: `{ref}`. A metric that moves by less than its repeat-to-repeat spread "
           "is noise.", "",
           "| Metric | " + " | ".join(f"`{m}`" for m in specs) + " |",
           "|---|" + "---|" * len(specs)]
    metrics = list(dict.fromkeys(list(MAX_DROP) + ["quality"]))
    for k in metrics:
        cells = []
        for m in specs:
            v = res[m]["metrics"].get(k)
            sp = res[m]["spread"].get(k, 0)
            cells.append("n/a" if v is None else
                         f"{v:.3f}" + (f" (±{sp / 2:.3f})" if report["repeats"] > 1 else ""))
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    out.append("| API errors | " + " | ".join(str(res[m]["errors"]) for m in specs) + " |")
    out.append("| seconds | " + " | ".join(str(res[m]["seconds"]) for m in specs) + " |")

    out += ["", f"## Would the regression gate allow swapping from `{ref}`?", ""]
    for m, v in report["verdicts"].items():
        verdict = "PASS" if v["passed"] else "BLOCKED on " + ", ".join(v["failed_metrics"])
        moved = [f"{k} {res[m]['metrics'][k] - res[ref]['metrics'][k]:+.2f}" for k in metrics
                 if k in res[m]["metrics"] and k in res[ref]["metrics"]
                 and abs(res[m]["metrics"][k] - res[ref]["metrics"][k]) >= NOTABLE]
        out.append(f"- `{m}`: **{verdict}**" + (f". Moved by {NOTABLE:.2f} or more: "
                                              + ", ".join(moved) if moved else ""))

    out += ["", "## Where each one went wrong", ""]
    for m in specs:
        misses = res[m]["misses"]
        out.append(f"### `{m}`")
        if not misses:
            out += ["No missed escalations, over-escalations or wrong categories.", ""]
            continue
        for kind, title in (("missed_escalation", "Missed escalations (a customer who needed a human did not get one)"),
                            ("over_escalation", "Over-escalations (sent to a human unnecessarily)"),
                            ("wrong_category", "Wrong category"),
                            ("error", "API errors")):
            lines = _miss_lines(misses, kind)
            if lines:
                out += [f"- **{title}:**"] + lines
        out.append("")

    if any(res[m]["errors"] for m in specs):
        out += ["> Some tickets hit API errors and count as failures. Fix that and rerun "
                "before drawing conclusions.", ""]
    if any(m.startswith("fake") for m in specs):
        out += ["> `fake` entries are offline stand-ins and say nothing about real quality.", ""]
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cx-compare", description=__doc__.split("\n\n")[0])
    p.add_argument("--models", nargs="+", required=True, metavar="SPEC",
                   help="2 or more of MODEL[@TRIAGE_PROMPT[/ESCALATION]], e.g. "
                        "ollama:llama3.2 or ollama:llama3.2@triage_v2/rules")
    p.add_argument("--prompt", default=prompts.DEFAULT_REPLY_PROMPT, help="reply prompt")
    p.add_argument("--triage-prompt", default=prompts.TRIAGE_PROMPT,
                   help="default triage prompt for entries without @...")
    p.add_argument("--escalation", choices=ESCALATION_MODES, default=prompts.DEFAULT_ESCALATION,
                   help="default escalation mode for entries without /...")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--cache", default=os.environ.get("CX_CACHE_DIR"), metavar="DIR")
    p.add_argument("--out", default="evals/compare", help="writes <out>.md and <out>.json")
    p.add_argument("--trace", action="store_true", help="send traces to LangSmith")
    args = p.parse_args(argv)

    if len(args.models) < 2 or args.repeats < 1:
        print("cx-compare: error: give at least 2 entries and --repeats >= 1", file=sys.stderr)
        return 2
    try:  # validate everything before spending any time or money
        prompts.load(args.prompt)
        parsed = [parse_spec(s, args.triage_prompt, args.escalation) for s in args.models]
        for model, _, _ in parsed:
            if (var := missing_key(model)):
                raise ValueError(f"{model} needs {var} to be set")
            get_model(model)
    except (ValueError, FileNotFoundError) as exc:
        print(f"cx-compare: error: {exc}", file=sys.stderr)
        return 2
    live = [m for m, _, _ in parsed if m.partition(":")[0] in LIVE_PROVIDERS]
    if live:
        from .dataset import load_tickets
        calls = len(load_tickets()) * CALLS_PER_TICKET * args.repeats
        note = "" if args.cache else " (add --cache DIR to make reruns free)"
        print(f"Up to ~{calls} calls per live entry{note}.", file=sys.stderr)
    try:
        with tracing_context(enabled=bool(args.trace)):
            report = run_compare(args.models, args.prompt, args.repeats, args.cache,
                                 args.triage_prompt, args.escalation)
    except Exception as exc:
        print(f"cx-compare: error: {exc}", file=sys.stderr)
        return 2

    md = render(report)
    print(md)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out.with_suffix('.md')} and {out.with_suffix('.json')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
