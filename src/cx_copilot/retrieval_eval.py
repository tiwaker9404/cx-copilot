"""Measure retrieval on its own, before any model is involved.

    cx-retrieval                                        # keyword, bm25, hybrid
    cx-retrieval --embed ollama:nomic-embed-text        # also embed and hybrid+embed (free, local)

For every ticket with labelled `relevant_policies` it asks each method for the top k
policies, using the GOLD category so retrieval is judged separately from triage:

    recall@1, recall@k   share of the relevant policies found in the top 1 / top k
    MRR                  1 / rank of the first relevant policy (0 if none found)
    precision@k          share of retrieved policies that are relevant
    no-policy noise      tickets with NO relevant policy where something was retrieved anyway
                         (the model then gets an irrelevant policy to quote)

Results go to evals/retrieval.md and evals/retrieval.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

from . import kb
from .dataset import load_tickets


def query_for(row: dict[str, Any]) -> str:
    return " ".join([row["ticket"]] + [m["text"] for m in row.get("history") or []
                                       if m.get("role") == "customer"])


def evaluate(retriever: kb.Retriever, rows: list[dict[str, Any]], k: int = 2) -> dict[str, Any]:
    r1, rk, mrr, prec, noise, misses = [], [], [], [], [], []
    for row in rows:
        gold = set(row["gold"]["relevant_policies"])
        found = [p.id for p in retriever.search(query_for(row), row["gold"]["category"], k)]
        if not gold:
            noise.append(float(bool(found)))
            continue
        r1.append(len(gold & set(found[:1])) / min(len(gold), 1))
        rk.append(len(gold & set(found)) / len(gold))
        rank = next((i for i, pid in enumerate(found, 1) if pid in gold), None)
        mrr.append(1 / rank if rank else 0.0)
        if found:
            prec.append(len(gold & set(found)) / len(found))
        if not gold <= set(found):
            misses.append({"id": row["id"], "want": sorted(gold), "got": found,
                           "ticket": row["ticket"][:90]})
    r = lambda v: round(fmean(v), 4) if v else None  # noqa: E731
    return {"method": retriever.name, "k": k, "labelled": len(r1),
            "recall@1": r(r1), f"recall@{k}": r(rk), "mrr": r(mrr), f"precision@{k}": r(prec),
            "no_policy_tickets": len(noise), "no_policy_noise": r(noise), "misses": misses}


def render(results: list[dict[str, Any]], k: int) -> str:
    cols = ["recall@1", f"recall@{k}", "mrr", f"precision@{k}", "no_policy_noise"]
    out = [f"# Retrieval eval: {results[0]['labelled']} tickets with a labelled policy, "
           f"{results[0]['no_policy_tickets']} with none, top {k}", "",
           "Gold category is used, so this measures retrieval alone. Lower no-policy noise is "
           "better.", "",
           "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for res in results:
        out.append(f"| `{res['method']}` | " + " | ".join(
            "n/a" if res[c] is None else f"{res[c]:.3f}" for c in cols) + " |")
    for res in results:
        if res["misses"]:
            out += ["", f"## `{res['method']}` missed ({len(res['misses'])})", ""]
            out += [f"- `{m['id']}` want {', '.join(m['want'])}; got "
                    f"{', '.join(m['got']) or 'nothing'}. \"{m['ticket']}\""
                    for m in res["misses"][:12]]
            if len(res["misses"]) > 12:
                out.append(f"- ...and {len(res['misses']) - 12} more in the JSON")
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cx-retrieval", description=__doc__.split("\n\n")[0])
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--methods", nargs="+", default=["keyword", "bm25", "hybrid"],
                   choices=kb.METHODS)
    p.add_argument("--embed", metavar="SPEC",
                   help="embedding model, e.g. ollama:nomic-embed-text; adds embed and "
                        "hybrid+embed")
    p.add_argument("--cache", default=os.environ.get("CX_CACHE_DIR"), metavar="DIR")
    p.add_argument("--out", default="evals/retrieval", help="writes <out>.md and <out>.json")
    args = p.parse_args(argv)
    rows = load_tickets()
    try:
        embedder = kb.get_embedder(args.embed, cache_dir=args.cache)
        retrievers = [kb.Retriever(method=m) for m in args.methods if m != "embed"]
        if embedder is not None:
            retrievers += [kb.Retriever(method="embed", embedder=embedder),
                           kb.Retriever(method="hybrid", embedder=embedder)]
        results = [evaluate(r, rows, args.k) for r in retrievers]
    except Exception as exc:
        print(f"cx-retrieval: error: {exc}", file=sys.stderr)
        return 2
    md = render(results, args.k)
    print(md)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
