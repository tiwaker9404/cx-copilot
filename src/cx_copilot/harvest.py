"""Turn what reviewers and customers told you into candidate eval cases.

    cx-harvest --db data/cx.db --out evals/candidates.jsonl

Pulls every reply a reviewer edited or rejected, and every reply rated <= --max-score, and
writes them in the tickets.jsonl format with the agent's own labels filled in as a starting
point. The labels are the MODEL's guesses, so every row is marked "needs_labels": true:

  1. open evals/candidates.jsonl and fix each row's gold labels (category, escalate,
     required_facts, needs_empathy, relevant_policies), then delete the extra fields
  2. append the rows to src/cx_copilot/data/tickets.jsonl
  3. run cx-gate; if the new cases lower the score, that is the point: they are the cases
     the agent gets wrong in production

Ticket text is already redacted (the service never stores raw PII). Ids already in the
dataset are skipped, so rerunning is safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .dataset import load_tickets
from .store import Store


def candidates(store: Store, max_score: float = 0.5,
               existing: set[str] | None = None) -> list[dict[str, Any]]:
    existing = existing or set()
    out: dict[str, dict[str, Any]] = {}
    for row in store.harvest_rows(max_score):
        cid = f"h_{row['ticket_id']}"
        if cid in existing:
            continue
        res = row["result"]
        cand = out.setdefault(cid, {
            "id": cid,
            "ticket": row["ticket"],
            "history": res.get("history") or [],
            "gold": {
                "category": res.get("category", "other"),
                "escalate": row["source"] == "review_reject" or bool(res.get("escalated")),
                "required_facts": [],
                "needs_empathy": res.get("sentiment") in ("angry", "frustrated"),
                "relevant_policies": list(res.get("policy_ids") or []),
            },
            "needs_labels": True,
            "sources": [],
            "notes": [],
        })
        if not cand["history"]:
            del cand["history"]
        cand["sources"].append(row["source"])
        if row.get("note"):
            cand["notes"].append(row["note"])
        if row["source"] == "review_edit":
            cand["agent_draft"] = row["draft"]
            cand["reference_reply"] = row["final_reply"]
        elif row["source"] == "review_reject":
            cand["agent_draft"] = row["draft"]
            cand["gold"]["escalate"] = True
        else:
            cand.setdefault("agent_reply", res.get("reply", ""))
            cand["feedback_score"] = row.get("score")
    return list(out.values())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cx-harvest", description=__doc__.split("\n\n")[0])
    p.add_argument("--db", default="data/cx.db")
    p.add_argument("--out", type=Path, default=Path("evals/candidates.jsonl"))
    p.add_argument("--max-score", type=float, default=0.5,
                   help="feedback at or below this counts as a bad reply")
    args = p.parse_args(argv)
    if not Path(args.db).exists():
        print(f"cx-harvest: error: no database at {args.db}", file=sys.stderr)
        return 2
    existing = {r["id"] for r in load_tickets()}
    rows = candidates(Store(args.db), args.max_score, existing)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    by_source: dict[str, int] = {}
    for r in rows:
        for s in r["sources"]:
            by_source[s] = by_source.get(s, 0) + 1
    print(f"Wrote {len(rows)} candidate cases to {args.out} "
          f"({', '.join(f'{k}: {v}' for k, v in sorted(by_source.items())) or 'none found'}). "
          "Fix their gold labels before adding them to the dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
