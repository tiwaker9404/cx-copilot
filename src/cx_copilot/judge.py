"""LLM-as-judge for reply quality, and the calibration that decides whether to believe it.

The deterministic scorers catch what a regex can catch. A judge model can read a reply the way
a team lead would ("did it actually answer the question?"), but a judge is just another model
that can be wrong. So before its scores are used for anything, `cx-judge` compares it with
human labels on a fixed set of replies (evals/judge_labels.jsonl) and reports agreement and
Cohen's kappa per criterion:

    cx-judge --model ollama:qwen2.5:7b --cache .cx_cache

A criterion is marked "trusted" only if kappa >= 0.6 and agreement >= 0.85. Use the judge in
an eval run with `cx-gate --judge MODEL` or `cx-compare --judge MODEL`; its scores show up as
judge_* metrics next to the deterministic ones.

Free: run the judge on a local model. Use a different model family from the agent when you
can, so the judge does not share the agent's blind spots.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from . import prompts
from .graph import format_history
from .models import ChatModel, get_model, missing_key, pop_usage
from .scorers import JUDGE_CRITERIA

JUDGE_PROMPT = "judge_v1"
DEFAULT_LABELS = Path("evals/judge_labels.jsonl")
TRUST_KAPPA = 0.6
TRUST_AGREEMENT = 0.85


def judge_user(ticket: str, history: list[dict[str, str]] | None, policies: list[str],
               reply: str) -> str:
    notes = "\n".join(f"- {p}" for p in policies) or "(no matching policy)"
    earlier = format_history(history) or "(none)"
    return (f"Customer's latest message:\n{ticket}\n\nEarlier conversation:\n{earlier}\n\n"
            f"Policy notes given to the agent:\n{notes}\n\nAgent reply:\n{reply}")


def parse_verdict(raw: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    out: dict[str, Any] = {}
    for key in JUDGE_CRITERIA:
        v = data.get(key)
        if isinstance(v, str):
            v = {"true": True, "false": False, "yes": True, "no": False}.get(v.strip().lower())
        out[key] = v if isinstance(v, bool) else None
    if all(v is None for v in out.values()):
        return None
    out["reason"] = str(data.get("reason", ""))[:300]
    return out


class Judge:
    def __init__(self, model: ChatModel, prompt: str = JUDGE_PROMPT) -> None:
        self.model = model
        self.prompt_name = prompt
        self.system = prompts.load(prompt)
        self.name = f"{model.name}/{prompt}"
        self.tokens = 0
        self.calls = 0

    def __call__(self, ticket: str, history: list[dict[str, str]] | None,
                 policies: list[str], reply: str) -> dict[str, Any] | None:
        """The verdict, or None if the judge returned something unusable."""
        context = {"ticket": ticket, "history": history or [], "policies": policies,
                   "reply": reply}
        pop_usage()
        raw = self.model.complete("judge", self.system,
                                  judge_user(ticket, history, policies, reply), context=context)
        usage = pop_usage()
        self.calls += 1
        if usage:
            self.tokens += usage["input_tokens"] + usage["output_tokens"]
        return parse_verdict(raw)


def get_judge(spec: str | None, *, cache_dir: str | None = None, prompt: str = JUDGE_PROMPT,
              client: Any = None) -> Judge | None:
    if not spec:
        return None
    return Judge(get_model(spec, cache_dir=cache_dir, client=client), prompt)


# ------------------------------------------------------------------ calibration
def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Agreement beyond chance between two raters. 1 = perfect, 0 = chance level."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return round((po - pe) / (1 - pe), 4)


def load_labels(path: str | Path = DEFAULT_LABELS) -> list[dict[str, Any]]:
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = {"id", "ticket", "policies", "reply", "labels"} - set(row)
        if missing:
            raise ValueError(f"{path}:{i}: missing {', '.join(sorted(missing))}")
        if set(row["labels"]) != set(JUDGE_CRITERIA):
            raise ValueError(f"{path}:{i}: labels must be exactly {', '.join(JUDGE_CRITERIA)}")
        rows.append(row)
    return rows


def calibrate(judge: Judge, labels: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts: dict[str, dict[str, Any] | None] = {}
    for row in labels:
        verdicts[row["id"]] = judge(row["ticket"], row.get("history"), row["policies"],
                                    row["reply"])
    criteria = {}
    disagreements = []
    for key in JUDGE_CRITERIA:
        human, model = [], []
        tp = fp = fn = tn = 0  # "positive" = the reply FAILS this criterion
        for row in labels:
            v = verdicts[row["id"]]
            if v is None or v.get(key) is None:
                continue
            h, m = bool(row["labels"][key]), bool(v[key])
            human.append(h)
            model.append(m)
            if not h and not m:
                tp += 1
            elif h and not m:
                fp += 1
            elif not h and m:
                fn += 1
            else:
                tn += 1
            if h != m:
                disagreements.append({"id": row["id"], "criterion": key, "human": h,
                                      "judge": m, "reason": v.get("reason", ""),
                                      "reply": row["reply"][:200]})
        n = len(human)
        agreement = round(sum(x == y for x, y in zip(human, model)) / n, 4) if n else None
        kappa = cohen_kappa(human, model)
        criteria[key] = {
            "n": n, "agreement": agreement, "kappa": kappa,
            "fails_caught": tp, "fails_missed": fn, "false_alarms": fp, "passes_agreed": tn,
            "trusted": bool(kappa is not None and agreement is not None
                            and kappa >= TRUST_KAPPA and agreement >= TRUST_AGREEMENT),
        }
    return {"judge": judge.name, "n": len(labels),
            "unparseable": sum(v is None for v in verdicts.values()),
            "criteria": criteria, "disagreements": disagreements,
            "calls": judge.calls, "tokens": judge.tokens}


def render_calibration(report: dict[str, Any]) -> str:
    out = [f"# Judge calibration: `{report['judge']}` on {report['n']} human-labelled replies",
           "",
           "\"Fail caught\" means the human said the reply fails the criterion and so did the "
           "judge. A criterion is trusted when kappa >= "
           f"{TRUST_KAPPA} and agreement >= {TRUST_AGREEMENT:.2f}.", "",
           "| Criterion | n | Agreement | Kappa | Fails caught | Fails missed | False alarms "
           "| Trusted |", "|---|---|---|---|---|---|---|---|"]
    for key, c in report["criteria"].items():
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        out.append(f"| {key} | {c['n']} | {fmt(c['agreement'])} | {fmt(c['kappa'])} | "
                   f"{c['fails_caught']} | {c['fails_missed']} | {c['false_alarms']} | "
                   f"{'yes' if c['trusted'] else 'no'} |")
    out += ["", f"Unparseable verdicts: {report['unparseable']} of {report['n']}. "
                f"Judge calls: {report['calls']}, tokens: {report['tokens']}.", ""]
    if report["disagreements"]:
        out += ["## Where the judge disagreed with the human labels", ""]
        for d in report["disagreements"][:15]:
            out.append(f"- `{d['id']}` {d['criterion']}: human {'pass' if d['human'] else 'FAIL'}"
                       f", judge {'pass' if d['judge'] else 'FAIL'}. Judge said: "
                       f"\"{d['reason'][:120]}\"")
        if len(report["disagreements"]) > 15:
            out.append(f"- ...and {len(report['disagreements']) - 15} more in the JSON")
    if report["judge"].startswith("fake"):
        out += ["", "> `fake` is the offline stand-in built from the same rules as the "
                    "labels' obvious cases. It shows the maths works, not that a judge does."]
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cx-judge", description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=os.environ.get("CX_JUDGE_MODEL", "fake"),
                   help="judge model, e.g. ollama:qwen2.5:7b")
    p.add_argument("--prompt", default=JUDGE_PROMPT)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--cache", default=os.environ.get("CX_CACHE_DIR"), metavar="DIR")
    p.add_argument("--out", default="evals/judge_calibration",
                   help="writes <out>.md and <out>.json")
    args = p.parse_args(argv)
    if (var := missing_key(args.model)):
        print(f"cx-judge: error: {args.model} needs {var} to be set", file=sys.stderr)
        return 2
    try:
        labels = load_labels(args.labels)
        judge = get_judge(args.model, cache_dir=args.cache, prompt=args.prompt)
        assert judge is not None
        report = calibrate(judge, labels)
    except Exception as exc:
        print(f"cx-judge: error: {exc}", file=sys.stderr)
        return 2
    md = render_calibration(report)
    print(md)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out.with_suffix('.md')} and {out.with_suffix('.json')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
