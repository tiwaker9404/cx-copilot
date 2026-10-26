"""The support agent as a LangGraph state machine.

    START -> classify --escalate--> escalate -> END
                 |
                 +--otherwise--> lookup_policy -> draft_reply -> qa_check --ok--> END
                                                      ^              |
                                                      +--- revise ---+   (bounded)

Design choices worth talking about:
  * Risky tickets (danger, legal, fraud, angry + urgent) never get a free-form LLM reply.
    They get a fixed holding message plus a handoff note for a human.
  * A regex safety net can force escalation even if the model says otherwise.
  * Unparseable triage output fails safe: escalate to a human.
  * The QA node bounces a bad draft back once; if it is still bad, we escalate.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from . import kb, prompts
from .models import CATEGORIES, SAFETY_RE, ChatModel
from .rules import quality_violations

MAX_ATTEMPTS = 2  # first draft + one revision

HOLDING_REPLY = (
    "I'm sorry about this. I've passed your case to a specialist on our senior team, "
    "who will contact you within 1 business day. - Northwind Support"
)


class TicketState(TypedDict, total=False):
    ticket: str
    triage: dict[str, Any]
    policy_ids: list[str]
    policies: list[str]
    reply: str
    handoff: str
    feedback: str
    qa_issues: list[str]
    attempts: int
    escalated: bool


_URGENCY = ("low", "normal", "high")
_SENTIMENT = ("angry", "frustrated", "neutral", "positive")
_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0", "none", "null", ""}
ESCALATION_MODES = ("model", "rules")


def _flag(value: Any) -> bool | None:
    """Read a yes/no field. The string "false" must not count as True."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def _label(value: Any, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    return text if text in allowed else default


def parse_triage(raw: str) -> dict[str, Any] | None:
    """Parse model output into a valid triage dict, or None if it is unusable.

    Labels are normalised ("High" -> "high", "safety/legal" -> "safety_legal"); unknown
    urgency or sentiment values fall back to the mild defaults. `escalate` is None when the
    model did not give a usable yes/no.
    """
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    category = _label(data.get("category"), CATEGORIES, "")
    if not category:
        return None
    return {
        "category": category,
        "urgency": _label(data.get("urgency"), _URGENCY, "normal"),
        "sentiment": _label(data.get("sentiment"), _SENTIMENT, "neutral"),
        "escalate": _flag(data.get("escalate")),
    }


def decide_escalation(triage: dict[str, Any]) -> bool:
    """The escalation policy, in code: the model labels, the rules decide."""
    return triage["category"] == "safety_legal" or (
        triage["sentiment"] == "angry" and triage["urgency"] == "high")


def build_graph(model: ChatModel, reply_prompt: str = prompts.DEFAULT_REPLY_PROMPT,
                policies: tuple[kb.Policy, ...] | None = None,
                triage_prompt: str = prompts.TRIAGE_PROMPT,
                escalation: str = prompts.DEFAULT_ESCALATION):
    """`escalation="model"` trusts the model's own escalate flag (triage_v1 asks for one);
    `"rules"` ignores it and applies `decide_escalation` to the labels. A prompt that does
    not ask for the flag (triage_v2) always uses the rules."""
    if escalation not in ESCALATION_MODES:
        raise ValueError(f"escalation must be one of {ESCALATION_MODES}, got {escalation!r}")
    reply_system = prompts.load(reply_prompt)
    triage_system = prompts.load(triage_prompt)
    meta = {"model": model.name, "reply_prompt": reply_prompt, "triage_prompt": triage_prompt,
            "escalation": escalation}

    @traceable(run_type="llm", name="model_call", metadata=meta)
    def call_model(task: str, system: str, user: str, context: dict[str, Any]) -> str:
        return model.complete(task, system, user, context=context)

    def classify(state: TicketState) -> dict[str, Any]:
        ticket = state["ticket"]
        triage = parse_triage(call_model("triage", triage_system, ticket, {"ticket": ticket}))
        if triage is None:  # fail safe
            triage = {"category": "other", "urgency": "high", "sentiment": "neutral",
                      "escalate": True}
        elif escalation == "rules" or triage["escalate"] is None:
            triage["escalate"] = decide_escalation(triage)
        if SAFETY_RE.search(ticket):  # hard rule beats the model
            triage["category"] = "safety_legal"
            triage["escalate"] = True
        return {"triage": triage, "attempts": 0, "escalated": False}

    def escalate(state: TicketState) -> dict[str, Any]:
        t = state["triage"]
        handoff = (f"[{t['category']} | urgency={t['urgency']} | sentiment={t['sentiment']}] "
                   f"Customer wrote: {state['ticket'][:400]}")
        return {"reply": HOLDING_REPLY, "handoff": handoff, "escalated": True}

    def lookup_policy(state: TicketState) -> dict[str, Any]:
        found = kb.retrieve(state["ticket"], state["triage"]["category"], policies=policies)
        return {"policy_ids": [p.id for p in found], "policies": [p.text for p in found]}

    def draft_reply(state: TicketState) -> dict[str, Any]:
        notes = "\n".join(f"- {p}" for p in state.get("policies", [])) or "(no matching policy)"
        user = (f"Ticket:\n{state['ticket']}\n\nTriage: {json.dumps(state['triage'])}\n\n"
                f"Policy notes:\n{notes}")
        if state.get("feedback"):
            user += f"\n\nYour previous draft was rejected: {state['feedback']}\nFix those issues."
        context = {"ticket": state["ticket"], "triage": state["triage"],
                   "policies": state.get("policies", []), "feedback": state.get("feedback", "")}
        reply = call_model("reply", reply_system, user, context).strip()
        return {"reply": reply, "attempts": state.get("attempts", 0) + 1}

    def qa_check(state: TicketState) -> dict[str, Any]:
        issues = quality_violations(state["reply"])
        out: dict[str, Any] = {"qa_issues": issues, "feedback": "; ".join(issues)}
        if issues and state.get("attempts", 0) >= MAX_ATTEMPTS:
            out.update(reply=HOLDING_REPLY, escalated=True,
                       handoff=f"QA failed after {MAX_ATTEMPTS} drafts: {'; '.join(issues)}")
        return out

    def after_classify(state: TicketState) -> str:
        return "escalate" if state["triage"]["escalate"] else "lookup_policy"

    def after_qa(state: TicketState) -> str:
        retry = bool(state.get("qa_issues")) and state.get("attempts", 0) < MAX_ATTEMPTS
        return "revise" if retry else "done"

    g = StateGraph(TicketState)
    g.add_node("classify", classify)
    g.add_node("escalate", escalate)
    g.add_node("lookup_policy", lookup_policy)
    g.add_node("draft_reply", draft_reply)
    g.add_node("qa_check", qa_check)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", after_classify,
                            {"escalate": "escalate", "lookup_policy": "lookup_policy"})
    g.add_edge("lookup_policy", "draft_reply")
    g.add_edge("draft_reply", "qa_check")
    g.add_conditional_edges("qa_check", after_qa, {"revise": "draft_reply", "done": END})
    g.add_edge("escalate", END)
    return g.compile()


def run_ticket(graph, ticket: str, *, model_name: str = "", reply_prompt: str = "",
               ticket_id: str = "") -> dict[str, Any]:
    """Invoke the graph with tags/metadata so runs are easy to filter in LangSmith."""
    config = {
        "run_name": "cx_copilot_ticket",
        "tags": ["cx-copilot", f"prompt:{reply_prompt}", f"model:{model_name}"],
        "metadata": {"ticket_id": ticket_id, "model": model_name, "reply_prompt": reply_prompt},
    }
    return graph.invoke({"ticket": ticket}, config=config)
