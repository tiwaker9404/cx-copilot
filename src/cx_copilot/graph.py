"""The support agent as a LangGraph state machine.

    START -> classify --escalate--> escalate -> END
                 |
                 +--otherwise--> lookup_policy -> draft_reply -> qa_check --ok--> [human_review] -> END
                                                      ^              |
                                                      +--- revise ---+   (bounded)

Design choices worth talking about:
  * Risky tickets (danger, legal, fraud, angry + urgent, a third contact about the same
    problem) never get a free-form LLM reply. They get a fixed holding message plus a
    handoff note for a human.
  * A regex safety net can force escalation even if the model says otherwise.
  * Unparseable triage output fails safe: escalate to a human.
  * The QA node checks format, safety AND commitments: every timeline, refund, credit,
    upgrade, escalation promise and menu path must be backed by the policy notes, and the
    reply must not ask for what the customer already gave. A bad draft is bounced back once
    with the reasons; if it is still bad, we escalate.
  * Optional human review: with review="risky" or "all", the graph pauses (LangGraph
    interrupt) before a reply leaves, and resumes with approve / edit / reject. The pause
    is checkpointed, so it survives a restart.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langsmith import traceable

from . import kb, prompts
from .models import CATEGORIES, SAFETY_RE, ChatModel, pop_usage
from .rules import commitment_violations, issue_kind, quality_violations

MAX_ATTEMPTS = 2  # first draft + one revision
REPEAT_CONTACT_LIMIT = 3  # the third customer message in a thread goes to a human
REVIEW_MODES = ("off", "risky", "all")

HOLDING_REPLY = (
    "I'm sorry about this. I've passed your case to a specialist on our senior team, "
    "who will contact you within 1 business day. - Northwind Support"
)


class TicketState(TypedDict, total=False):
    ticket: str
    history: list[dict[str, str]]
    triage: dict[str, Any]
    policy_ids: list[str]
    policies: list[str]
    reply: str
    handoff: str
    feedback: str
    qa_issues: list[str]
    first_issues: list[str]
    first_draft: str
    attempts: int
    escalated: bool
    escalation_reason: str
    review_reason: str
    review: dict[str, Any]
    usage: dict[str, Any]


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


def customer_contacts(history: list[dict[str, str]] | None) -> int:
    """How many times the customer has written, counting the current message."""
    return 1 + sum(1 for m in history or [] if m.get("role") == "customer")


def format_history(history: list[dict[str, str]] | None) -> str:
    return "\n".join(f"{'Customer' if m.get('role') == 'customer' else 'Agent'}: {m['text']}"
                      for m in history or [])


def customer_text(state: TicketState) -> str:
    """Everything the customer has written in this conversation."""
    return " ".join([state["ticket"]] + [m["text"] for m in state.get("history") or []
                                         if m.get("role") == "customer"])


def review_reason(state: TicketState, mode: str) -> str:
    """Why this auto-reply should be checked by a person before it is sent ("" = no need)."""
    if mode == "off":
        return ""
    if mode == "all":
        return "review mode is 'all'"
    t = state["triage"]
    reasons = []
    if state.get("first_issues"):
        kinds = sorted({issue_kind(i) for i in state["first_issues"]})
        reasons.append("first draft failed QA (" + ", ".join(kinds) + ")")
    if t.get("sentiment") in ("angry", "frustrated"):
        reasons.append(f"customer is {t['sentiment']}")
    if t.get("urgency") == "high":
        reasons.append("high urgency")
    if state.get("history"):
        reasons.append("follow-up in an ongoing conversation")
    if not state.get("policy_ids"):
        reasons.append("no matching policy")
    return "; ".join(reasons)


def _add_usage(state: TicketState, usage: dict[str, Any] | None) -> dict[str, Any]:
    total = dict(state.get("usage") or {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                        "model_ms": 0, "latency_known": True})
    total["calls"] += 1
    if usage:
        total["input_tokens"] += usage["input_tokens"]
        total["output_tokens"] += usage["output_tokens"]
        if usage.get("latency_ms") is None:
            total["latency_known"] = False
        else:
            total["model_ms"] += usage["latency_ms"]
    return total


def build_graph(model: ChatModel, reply_prompt: str = prompts.DEFAULT_REPLY_PROMPT,
                policies: tuple[kb.Policy, ...] | None = None,
                triage_prompt: str = prompts.TRIAGE_PROMPT,
                escalation: str = prompts.DEFAULT_ESCALATION, *,
                retriever: kb.Retriever | None = None, k: int = 2,
                review: str = "off", checkpointer: Any = None):
    """`escalation="model"` trusts the model's own escalate flag (triage_v1 asks for one);
    `"rules"` ignores it and applies `decide_escalation` to the labels. A prompt that does
    not ask for the flag (triage_v2) always uses the rules.

    `review` pauses risky (or all) auto-replies for a person; it needs a `checkpointer`.
    """
    if escalation not in ESCALATION_MODES:
        raise ValueError(f"escalation must be one of {ESCALATION_MODES}, got {escalation!r}")
    if review not in REVIEW_MODES:
        raise ValueError(f"review must be one of {REVIEW_MODES}, got {review!r}")
    if review != "off" and checkpointer is None:
        raise ValueError("human review needs a checkpointer so a paused ticket can resume")
    retriever = retriever or kb.Retriever(policies)
    reply_system = prompts.load(reply_prompt)
    triage_system = prompts.load(triage_prompt)
    meta = {"model": model.name, "reply_prompt": reply_prompt, "triage_prompt": triage_prompt,
            "escalation": escalation, "retrieval": retriever.name}

    @traceable(run_type="llm", name="model_call", metadata=meta)
    def call_model(task: str, system: str, user: str, context: dict[str, Any]) -> str:
        return model.complete(task, system, user, context=context)

    def _call(state: TicketState, task: str, system: str, user: str,
              context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        pop_usage()
        text = call_model(task, system, user, context)
        return text, _add_usage(state, pop_usage())

    def classify(state: TicketState) -> dict[str, Any]:
        ticket, history = state["ticket"], state.get("history") or []
        user = ticket if not history else (
            f"Earlier in this conversation:\n{format_history(history)}\n\n"
            f"Latest customer message:\n{ticket}")
        raw, usage = _call(state, "triage", triage_system, user,
                           {"ticket": ticket, "history": history})
        triage = parse_triage(raw)
        reason = "triage"
        if triage is None:  # fail safe
            triage = {"category": "other", "urgency": "high", "sentiment": "neutral",
                      "escalate": True}
            reason = "triage_unparseable"
        elif escalation == "rules" or triage["escalate"] is None:
            triage["escalate"] = decide_escalation(triage)
        if SAFETY_RE.search(customer_text(state)):  # hard rule beats the model
            triage["category"] = "safety_legal"
            triage["escalate"] = True
            reason = "safety_rule"
        elif customer_contacts(history) >= REPEAT_CONTACT_LIMIT:  # the bot already had its chances
            triage["escalate"] = True
            reason = "repeat_contact"
        return {"triage": triage, "attempts": 0, "escalated": False, "usage": usage,
                "escalation_reason": reason if triage["escalate"] else ""}

    def escalate(state: TicketState) -> dict[str, Any]:
        t = state["triage"]
        thread = (f" Earlier messages: {format_history(state.get('history'))[:600]}"
                  if state.get("history") else "")
        handoff = (f"[{t['category']} | urgency={t['urgency']} | sentiment={t['sentiment']} | "
                   f"reason={state.get('escalation_reason') or 'triage'}] "
                   f"Customer wrote: {state['ticket'][:400]}{thread}")
        return {"reply": HOLDING_REPLY, "handoff": handoff, "escalated": True}

    def lookup_policy(state: TicketState) -> dict[str, Any]:
        found = retriever.search(customer_text(state), state["triage"]["category"], k)
        return {"policy_ids": [p.id for p in found], "policies": [p.text for p in found]}

    def draft_reply(state: TicketState) -> dict[str, Any]:
        notes = "\n".join(f"- {p}" for p in state.get("policies", [])) or "(no matching policy)"
        history = state.get("history") or []
        user = (f"Earlier conversation:\n{format_history(history)}\n\n" if history else "")
        user += (f"Ticket:\n{state['ticket']}\n\nTriage: {json.dumps(state['triage'])}\n\n"
                 f"Policy notes:\n{notes}")
        if state.get("feedback"):
            user += f"\n\nYour previous draft was rejected: {state['feedback']}\nFix those issues."
        context = {"ticket": state["ticket"], "triage": state["triage"], "history": history,
                   "policies": state.get("policies", []), "feedback": state.get("feedback", "")}
        reply, usage = _call(state, "reply", reply_system, user, context)
        return {"reply": reply.strip(), "attempts": state.get("attempts", 0) + 1,
                "usage": usage}

    def qa_check(state: TicketState) -> dict[str, Any]:
        issues = quality_violations(state["reply"]) + commitment_violations(
            state["reply"], state.get("policies", []), customer_text(state))
        out: dict[str, Any] = {"qa_issues": issues, "feedback": "; ".join(issues)}
        if state.get("attempts", 0) == 1:
            out["first_issues"] = issues
            out["first_draft"] = state["reply"]
        if issues and state.get("attempts", 0) >= MAX_ATTEMPTS:
            out.update(reply=HOLDING_REPLY, escalated=True, escalation_reason="qa",
                       handoff=f"QA failed after {MAX_ATTEMPTS} drafts: {'; '.join(issues)}. "
                               f"Customer wrote: {state['ticket'][:400]}")
        elif not issues:
            out["review_reason"] = review_reason({**state, **out}, review)
        return out

    def human_review(state: TicketState) -> dict[str, Any]:
        decision = interrupt({"reply": state["reply"], "reason": state["review_reason"],
                              "ticket": state["ticket"], "triage": state["triage"],
                              "policy_ids": state.get("policy_ids", [])})
        action = decision.get("action")
        note = str(decision.get("note") or "")
        record = {"action": action, "note": note, "draft": state["reply"]}
        if action == "approve":
            return {"review": record}
        if action == "edit":
            return {"reply": str(decision["reply"]).strip(), "review": record}
        if action == "reject":
            return {"review": record, "reply": HOLDING_REPLY, "escalated": True,
                    "escalation_reason": "review_rejected",
                    "handoff": f"Draft rejected in review ({note or 'no note'}). "
                               f"Customer wrote: {state['ticket'][:400]}"}
        raise ValueError(f"unknown review action {action!r}")

    def after_classify(state: TicketState) -> str:
        return "escalate" if state["triage"]["escalate"] else "lookup_policy"

    def after_qa(state: TicketState) -> str:
        if state.get("qa_issues") and state.get("attempts", 0) < MAX_ATTEMPTS:
            return "revise"
        if not state.get("escalated") and state.get("review_reason"):
            return "review"
        return "done"

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
    routes = {"revise": "draft_reply", "done": END}
    if review != "off":
        g.add_node("human_review", human_review)
        g.add_edge("human_review", END)
        routes["review"] = "human_review"
    else:
        routes["review"] = END
    g.add_conditional_edges("qa_check", after_qa, routes)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=checkpointer)


def run_ticket(graph, ticket: str, *, history: list[dict[str, str]] | None = None,
               model_name: str = "", reply_prompt: str = "", triage_prompt: str = "",
               escalation: str = "", retrieval: str = "", ticket_id: str = "",
               thread_id: str | None = None) -> dict[str, Any]:
    """Invoke the graph with tags/metadata so runs are easy to filter in LangSmith."""
    tags = ["cx-copilot", f"prompt:{reply_prompt}", f"model:{model_name}"]
    tags += [f"{k}:{v}" for k, v in (("triage", triage_prompt), ("escalation", escalation),
                                     ("retrieval", retrieval)) if v]
    if history:
        tags.append("multi-turn")
    config: dict[str, Any] = {
        "run_name": "cx_copilot_ticket",
        "tags": tags,
        "metadata": {"ticket_id": ticket_id, "model": model_name, "reply_prompt": reply_prompt,
                     "triage_prompt": triage_prompt, "escalation": escalation,
                     "retrieval": retrieval, "turns": customer_contacts(history)},
    }
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    return graph.invoke({"ticket": ticket, "history": list(history or [])}, config=config)
