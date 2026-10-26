"""Evaluators for support replies.

Every scorer has the signature `(outputs, reference_outputs) -> {"key", "score"}` so the same
functions work in the offline runner AND as `evaluators=[...]` in `langsmith.evaluate`.
A score of `None` means "not applicable to this ticket" and is excluded from averages.

These are deterministic heuristics on purpose: they are free, fast, repeatable, and safe to
run in CI. They are blunt instruments (see the README's "Limits"), which is why the next
step in the series is an LLM judge calibrated against them.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .rules import MAX_WORDS, MIN_WORDS, safety_violations, word_count

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


COMPONENTS: tuple[Scorer, ...] = (
    category_accuracy, escalation_correct, escalation_recall, policy_grounded,
    empathy, actionable, length_ok, safety, first_try_clean,
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


ALL_SCORERS: tuple[Scorer, ...] = (*COMPONENTS, quality)
