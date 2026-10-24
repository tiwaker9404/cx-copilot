"""Model adapters.

`FakeModel` is a deterministic stand-in so the whole project (graph, evals, regression
gate, CI) runs offline with no API keys. It is NOT a quality benchmark: it reads simple
directives out of the prompt text, so changing the prompt changes its behaviour in a
predictable way. That is exactly what you need to test the harness. For real quality
numbers, run with a live model.

Spec strings accepted by `get_model`:
    fake                      offline stand-in (default)
    fake:lite                 offline stand-in that ignores urgency/sentiment (a "cheaper model")
    anthropic:<model-name>    needs ANTHROPIC_API_KEY and `pip install -e ".[live]"`
    openai:<model-name>       needs OPENAI_API_KEY and `pip install -e ".[live]"`
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

CATEGORIES = ("billing", "returns", "shipping", "account", "technical", "safety_legal", "other")

# Hard safety net used by the graph AND by the fake model. Word-bounded on purpose.
SAFETY_RE = re.compile(
    r"\b(on fire|caught fire|smok(?:e|ing)|burn(?:ed|t|ing)?|overheat\w*|sparks?|"
    r"electric shock|injur\w+|lawyer|attorney|sue|legal action|lawsuit|fraud\w*|"
    r"unauthori[sz]ed|hacked)\b",
    re.IGNORECASE,
)

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    # Order matters: it breaks ties.
    "returns": ("refund", "return", "money back", "exchange", "defective", "broken",
                "damaged", "cracked", "warranty"),
    "billing": ("charged", "charge", "invoice", "billing", "subscription", "cancel",
                "renew", "price", "payment"),
    "shipping": ("shipping", "ship ", "delivery", "delivered", "package", "arrive",
                 "tracking", "order is", "late"),
    "account": ("password", "log in", "login", "sign in", "locked out", "account", "email"),
    "technical": ("crash", "error", "bug", "app", "freez", "won't open", "not loading",
                  "website"),
}
_ANGRY = ("furious", "unacceptable", "ridiculous", "worst", "disgusting", "fed up", "outraged")
_FRUSTRATED = ("frustrat", "annoyed", "angry", "terrible", "upset", "disappointed")
_POSITIVE = ("thanks", "thank you", "love", "amazing", "great", "appreciate")
_URGENT = ("urgent", "asap", "immediately", "right now", "today", "in an hour", "emergency")
_RELAXED = ("no rush", "whenever", "just curious")


class ChatModel(Protocol):
    name: str

    def complete(self, task: str, system: str, user: str, *,
                 context: dict[str, Any] | None = None) -> str:
        """Return the model's text. `task` is "triage" or "reply".

        `context` carries structured inputs for the offline stand-in; live adapters
        ignore it because `user` already contains everything they need.
        """
        ...


def keyword_triage(ticket: str, *, lite: bool = False) -> dict[str, Any]:
    text = ticket.lower()
    if SAFETY_RE.search(ticket):
        category = "safety_legal"
    else:
        counts = {c: sum(1 for kw in kws if kw in text) for c, kws in _CATEGORY_KEYWORDS.items()}
        best = max(counts.values())
        category = next(c for c, n in counts.items() if n == best) if best > 0 else "other"

    if lite:  # the "cheaper model" never looks at tone or urgency
        urgency, sentiment = "normal", "neutral"
    else:
        urgency = ("high" if any(w in text for w in _URGENT) or category == "safety_legal"
                   else "low" if any(w in text for w in _RELAXED) else "normal")
        sentiment = ("angry" if any(w in text for w in _ANGRY)
                     else "frustrated" if any(w in text for w in _FRUSTRATED)
                     else "positive" if any(w in text for w in _POSITIVE) else "neutral")
    escalate = category == "safety_legal" or (sentiment == "angry" and urgency == "high")
    return {"category": category, "urgency": urgency, "sentiment": sentiment,
            "escalate": escalate}


_FORBIDDEN_SENTENCE = "We guarantee this will be fixed today, no exceptions."
_FILLER = ("We truly value every single customer and we are always working hard to improve "
           "every part of the shopping experience, from the first click to the final "
           "delivery, so that every order feels special and every question gets the "
           "attention it deserves from our whole team here at Northwind. ") * 2

_NEXT_STEP = {
    "returns": "Next step: reply with your order number and we'll start the process.",
    "billing": "Next step: reply with your order number so we can review the charge.",
    "shipping": "Next step: reply with your order number and we'll check the carrier status.",
    "account": "Next step: follow the steps above and reply if you're still stuck.",
    "technical": "Next step: please reply with your device model and OS version.",
    "other": "Next step: reply with a few more details and we'll point you in the right direction.",
}


class FakeModel:
    """Deterministic stand-in. Behaviour is driven by directives found in the prompt."""

    def __init__(self, variant: str = "default") -> None:
        self.variant = variant
        self.name = "fake" if variant == "default" else f"fake:{variant}"

    def complete(self, task: str, system: str, user: str, *,
                 context: dict[str, Any] | None = None) -> str:
        context = context or {}
        if task == "triage":
            return json.dumps(keyword_triage(context.get("ticket", user),
                                             lite=self.variant == "lite"))
        if task == "reply":
            return self._reply(system.lower(), context)
        raise ValueError(f"unknown task {task!r}")

    def _reply(self, system: str, ctx: dict[str, Any]) -> str:
        triage = ctx.get("triage", {})
        policies: list[str] = ctx.get("policies", [])
        clean_mode = bool(ctx.get("feedback"))  # a QA-driven revision

        acknowledge = "acknowledge the customer" in system
        cite = "quote the exact policy" in system
        next_step = "end with one clear next step" in system
        no_promises = "never promise" in system
        cap_match = re.search(r"at most (\d+) words", system)
        max_words = int(cap_match.group(1)) if cap_match else None

        parts: list[str] = []
        if acknowledge:
            if triage.get("sentiment") == "positive":
                parts.append("Thanks so much for the kind words and for reaching out.")
            else:
                parts.append("I'm sorry for the trouble, and I understand how frustrating "
                             "this can be.")
        else:
            parts.append("Thanks for contacting us.")

        if policies and cite:
            parts.append(" ".join(policies))
        elif policies:
            parts.append("We'll look into this and get it sorted for you.")

        if next_step:
            parts.append(_NEXT_STEP.get(triage.get("category", "other"), _NEXT_STEP["other"]))

        if not no_promises and not clean_mode:
            parts.append(_FORBIDDEN_SENTENCE)
        if max_words is None and not clean_mode:
            parts.append(_FILLER.strip())

        parts.append("- Northwind Support")
        reply = " ".join(parts)
        if max_words and len(reply.split()) > max_words and len(policies) > 1 and cite:
            parts[1] = policies[0]
            reply = " ".join(parts)
        return reply


class AnthropicModel:
    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - needs optional extra
            raise RuntimeError('Install the live extras: pip install -e ".[live]"') from exc
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._model = model
        self.name = f"anthropic:{model}"

    def complete(self, task: str, system: str, user: str, *,
                 context: dict[str, Any] | None = None) -> str:  # pragma: no cover - live
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class OpenAIModel:
    def __init__(self, model: str) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - needs optional extra
            raise RuntimeError('Install the live extras: pip install -e ".[live]"') from exc
        self._client = openai.OpenAI()  # reads OPENAI_API_KEY
        self._model = model
        self.name = f"openai:{model}"

    def complete(self, task: str, system: str, user: str, *,
                 context: dict[str, Any] | None = None) -> str:  # pragma: no cover - live
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


def get_model(spec: str = "fake") -> ChatModel:
    provider, _, rest = spec.partition(":")
    if provider == "fake":
        return FakeModel(rest or "default")
    if provider == "anthropic" and rest:
        return AnthropicModel(rest)
    if provider == "openai" and rest:
        return OpenAIModel(rest)
    raise ValueError(f"Bad model spec {spec!r}. Try 'fake', 'fake:lite', "
                     "'anthropic:<model>' or 'openai:<model>'.")
