"""Evaluators for support replies.

Every scorer has the signature `(outputs, reference_outputs) -> {"key", "score"}` so the same
functions work in the offline runner AND as `evaluators=[...]` in `langsmith.evaluate`.
A score of `None` means "not applicable to this ticket" and is excluded from averages.

These are deterministic heuristics on purpose: they are free, fast, repeatable, and safe to
run in CI. They are blunt instruments (see the README's "Limits"), so `judge.py` adds an
optional LLM judge (judge_* metrics), calibrated against human labels before it is trusted.

Outputs carry `ticket`, `history` and `policies` too, so the commitment checks can see what
the reply was allowed to say.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .rules import (INVENTED_KINDS, MAX_WORDS, MIN_WORDS, commitment_violations, issue_kind,
                    safety_violations, word_count)

Scorer = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]

_EMPATHY = ("sorry", "apolog", "understand", "frustrat", "i can imagine", "that sounds")
_ACTION = ("next step", "please reply", "reply with", "reply to", "go to", "visit", "click",
           "tap", "you can", "send us", "share", "let us know", "contact us", "try ", "open ")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _result(key: str, score: float | None) -> dict[str, Any]:
    return {"key": key, "score": score}


def category_accuracy(outputs, reference_outputs):
    return _result("category_accuracy",
                   float(outputs["triage"]["category"] == reference_outputs["category"]))


def escalation_correct(outputs, reference_outputs):
    return _result("escalation_correct",
                   float(bool(outputs["escalated"]) == bool(reference_outputs["escalate"])))


def escalation_recall(outputs, reference_outputs):
    """Only tickets that SHOULD escalate count. Missing one is the costly mistake."""
    if not reference_outputs["escalate"]:
        return _result("escalation_recall", None)
    return _result("escalation_recall", float(bool(outputs["escalated"])))


def policy_grounded(outputs, reference_outputs):
    facts = reference_outputs.get("required_facts") or []
    if not facts or outputs["escalated"]:
        return _result("policy_grounded", None)
    reply = _norm(outputs["reply"])
    return _result("policy_grounded", sum(_norm(f) in reply for f in facts) / len(facts))


def empathy(outputs, reference_outputs):
    if not reference_outputs.get("needs_empathy"):
        return _result("empathy", None)
    reply = outputs["reply"].lower()
    return _result("empathy", float(any(w in reply for w in _EMPATHY)))


def actionable(outputs, reference_outputs):
    if outputs["escalated"]:
        return _result("actionable", None)
    reply = outputs["reply"].lower()
    return _result("actionable", float(any(w in reply for w in _ACTION)))


def length_ok(outputs, reference_outputs):
    n = word_count(outputs["reply"])
    return _result("length_ok", float(MIN_WORDS <= n <= MAX_WORDS))


def safety(outputs, reference_outputs):
    return _result("safety", float(not safety_violations(outputs["reply"])))


def first_try_clean(outputs, reference_outputs):
    """Did the first draft pass QA? Efficiency signal: retries cost tokens and latency."""
    if outputs.get("attempts", 0) == 0:
        return _result("first_try_clean", None)
    return _result("first_try_clean", float(outputs["attempts"] == 1))


def _customer_text(outputs) -> str:
    return " ".join([outputs.get("ticket", "")] + [m["text"] for m in outputs.get("history") or []
                                                   if m.get("role") == "customer"])


def _issues(outputs) -> list[str]:
    return commitment_violations(outputs["reply"], outputs.get("policies") or [],
                                 _customer_text(outputs))


def no_invented_commitments(outputs, reference_outputs):
    """The reply that went out: no timeline, refund, credit, upgrade, escalation promise or
    menu path that the policy notes do not back. With the QA guardrail this should be 1.0;
    anything less means the guardrail let one through."""
    if outputs["escalated"]:
        return _result("no_invented_commitments", None)
    bad = any(issue_kind(i) in INVENTED_KINDS for i in _issues(outputs))
    return _result("no_invented_commitments", float(not bad))


def no_repeat_questions(outputs, reference_outputs):
    """Does not ask for an order number or email the customer already gave."""
    if outputs["escalated"]:
        return _result("no_repeat_questions", None)
    bad = any(issue_kind(i) == "repeat question" for i in _issues(outputs))
    return _result("no_repeat_questions", float(not bad))


def draft_commitment_free(outputs, reference_outputs):
    """The MODEL's behaviour before the guardrail: did its first draft invent anything?
    This is the number that shows whether a prompt or model makes things up."""
    if outputs.get("attempts", 0) == 0:
        return _result("draft_commitment_free", None)
    first = outputs.get("first_issues") or []
    return _result("draft_commitment_free",
                   float(not any(issue_kind(i) in INVENTED_KINDS for i in first)))


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def followup_not_repeated(outputs, reference_outputs):
    """In a thread, the new reply must not be a near copy of the agent's previous one."""
    agent = [m["text"] for m in outputs.get("history") or [] if m.get("role") == "agent"]
    if not agent or outputs["escalated"]:
        return _result("followup_not_repeated", None)
    a, b = _words(outputs["reply"]), _words(agent[-1])
    jaccard = len(a & b) / len(a | b) if a | b else 1.0
    return _result("followup_not_repeated", float(jaccard < 0.7))


def retrieval_recall(outputs, reference_outputs):
    """Share of the labelled relevant policies that retrieval found (in the live pipeline,
    with the model's own category)."""
    gold = reference_outputs.get("relevant_policies") or []
    if not gold or outputs["escalated"]:
        return _result("retrieval_recall", None)
    found = set(outputs.get("policy_ids") or [])
    return _result("retrieval_recall", sum(p in found for p in gold) / len(gold))


def _judge(key: str) -> Scorer:
    def scorer(outputs, reference_outputs):
        verdict = outputs.get("judge")
        if not verdict or outputs["escalated"] or verdict.get(key) is None:
            return _result(f"judge_{key}", None)
        return _result(f"judge_{key}", float(bool(verdict[key])))
    scorer.__name__ = f"judge_{key}"
    return scorer


JUDGE_CRITERIA = ("grounded", "answers_question", "tone_ok", "no_invented_commitments")
JUDGE_SCORERS: tuple[Scorer, ...] = tuple(_judge(k) for k in JUDGE_CRITERIA)

COMPONENTS: tuple[Scorer, ...] = (
    category_accuracy, escalation_correct, escalation_recall, policy_grounded,
    empathy, actionable, length_ok, safety, first_try_clean, no_invented_commitments,
    no_repeat_questions, draft_commitment_free, followup_not_repeated, retrieval_recall,
)

# Weights for the composite. Renormalised per ticket over the metrics that apply.
WEIGHTS = {
    "category_accuracy": 0.15,
    "escalation_correct": 0.15,
    "policy_grounded": 0.25,
    "empathy": 0.15,
    "actionable": 0.10,
    "length_ok": 0.05,
    "safety": 0.15,
    "no_invented_commitments": 0.10,
    "no_repeat_questions": 0.05,
    "followup_not_repeated": 0.05,
}


def quality(outputs, reference_outputs):
    """Weighted per-ticket composite over the metrics that apply to this ticket."""
    num = den = 0.0
    for scorer in COMPONENTS:
        res = scorer(outputs, reference_outputs)
        weight = WEIGHTS.get(res["key"])
        if weight is None or res["score"] is None:
            continue
        num += weight * res["score"]
        den += weight
    return _result("quality", num / den if den else None)


ALL_SCORERS: tuple[Scorer, ...] = (*COMPONENTS, quality, *JUDGE_SCORERS)
