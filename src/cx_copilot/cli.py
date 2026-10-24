"""Try the agent on one ticket:  cx-copilot "I was charged twice for my order" """

from __future__ import annotations

import argparse
import json
import os
import sys

from langsmith import tracing_context

from . import prompts
from .graph import build_graph, run_ticket
from .models import get_model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cx-copilot", description=__doc__)
    p.add_argument("ticket", help="the customer message")
    p.add_argument("--prompt", default=prompts.DEFAULT_REPLY_PROMPT)
    p.add_argument("--model", default=os.environ.get("CX_MODEL", "fake"))
    p.add_argument("--trace", action="store_true", help="send the trace to LangSmith")
    args = p.parse_args(argv)

    model = get_model(args.model)
    graph = build_graph(model, args.prompt)
    with tracing_context(enabled=bool(args.trace)):
        state = run_ticket(graph, args.ticket, model_name=model.name, reply_prompt=args.prompt)

    print("triage :", json.dumps(state["triage"]))
    print("policy :", ", ".join(state.get("policy_ids", [])) or "-")
    print("route  :", "ESCALATED to a human" if state.get("escalated") else "auto-reply")
    print("\n" + state["reply"])
    if state.get("handoff"):
        print("\nhandoff note:", state["handoff"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
