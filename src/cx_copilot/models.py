"""Model adapters.

`FakeModel` is a deterministic stand-in so the whole project (graph, evals, regression
gate, CI) runs offline with no API keys. It is NOT a quality benchmark: it reads simple
directives out of the prompt text, so changing the prompt changes its behaviour in a
predictable way. That is exactly what you need to test the harness. For real quality
numbers, run with a live model.

Spec strings accepted by `get_model`:
    fake                      offline stand-in (default)
    fake:lite                 offline stand-in that ignores urgency/sentiment (a "cheaper model")
    ollama:<model>            free local model via Ollama (OLLAMA_BASE_URL, default localhost)
    openai_compat:<model>     any OpenAI-compatible server (CX_OPENAI_COMPAT_BASE_URL)
    anthropic:<model-name>    needs ANTHROPIC_API_KEY and `pip install -e ".[live]"`
    openai:<model-name>       needs OPENAI_API_KEY and `pip install -e ".[live]"`
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
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
    "shipping": ("shipping", "ship ", "shipped", "delivery", "delivered", "package", "arrive",
                 "tracking", "order is", "late"),
    "account": ("password", "log in", "login", "sign in", "locked out", "account", "email"),
    "technical": ("crash", "error", "bug", "app", "freez", "won't open", "not loading",
                  "website"),
}
_ANGRY = ("furious", "unacceptable", "ridiculous", "worst", "disgusting", "fed up", "outraged")
_FRUSTRATED = ("frustrat", "annoyed", "angry", "terrible", "upset", "disappointed")
_POSITIVE = ("thanks", "thank you", "love", "amazing", "great", "appreciate")
_URGENT = ("urgent", "asap", "immediately", "right now", "today", "tomorrow", "in an hour",
           "emergency")
_RELAXED = ("no rush", "whenever", "just curious")


# ------------------------------------------------------------------ usage
# Each adapter records the tokens and latency of its last call here. Thread-local, because the
# service runs several tickets at once on one model object.
_usage = threading.local()


def estimate_tokens(*texts: str) -> int:
    """About 4 characters per token. Used when a server reports no usage (and by `fake`)."""
    return sum(len(t) for t in texts) // 4 + 1


def record_usage(input_tokens: int, output_tokens: int, latency_ms: int | None, *,
                 cached: bool = False, estimated: bool = False) -> None:
    _usage.last = {"input_tokens": int(input_tokens), "output_tokens": int(output_tokens),
                   "latency_ms": latency_ms, "cached": cached, "estimated": estimated}


def pop_usage() -> dict[str, Any] | None:
    """The usage of the last call on this thread, then clear it."""
    last = getattr(_usage, "last", None)
    _usage.last = None
    return last


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
            out = json.dumps(triage_thread(context.get("ticket", user),
                                           context.get("history") or [],
                                           lite=self.variant == "lite"))
        elif task == "reply":
            out = self._reply(system.lower(), context)
        elif task == "judge":
            out = json.dumps(fake_judge(context))
        else:
            raise ValueError(f"unknown task {task!r}")
        record_usage(estimate_tokens(system, user), estimate_tokens(out), 0, estimated=True)
        return out

    def _reply(self, system: str, ctx: dict[str, Any]) -> str:
        from .rules import ORDER_GIVEN_RE

        triage = ctx.get("triage", {})
        policies: list[str] = ctx.get("policies", [])
        history = ctx.get("history") or []
        clean_mode = bool(ctx.get("feedback"))  # a QA-driven revision
        customer_text = " ".join([ctx.get("ticket", "")] + [m["text"] for m in history
                                                             if m.get("role") == "customer"])

        acknowledge = "acknowledge the customer" in system
        cite = "quote the exact policy" in system
        next_step = "end with one clear next step" in system
        no_promises = "never promise" in system
        grounded = "only offer what the policy notes say" in system
        no_repeat = "already gave" in system
        thread_aware = "earlier conversation" in system
        cap_match = re.search(r"at most (\d+) words", system)
        max_words = int(cap_match.group(1)) if cap_match else None

        parts: list[str] = []
        if history and thread_aware:
            parts.append("Thanks for following up.")
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
            if (no_repeat or clean_mode) and ORDER_GIVEN_RE.search(customer_text):
                parts.append(_NEXT_STEP_KNOWN_ORDER)
            else:
                parts.append(_NEXT_STEP.get(triage.get("category", "other"),
                                            _NEXT_STEP["other"]))

        # Like real models, an ungrounded prompt over-reassures upset customers.
        if (not grounded and not clean_mode
                and triage.get("sentiment") in ("frustrated", "angry")):
            parts.append(_INVENTED_COMMITMENT)
        if not no_promises and not clean_mode:
            parts.append(_FORBIDDEN_SENTENCE)
        if max_words is None and not clean_mode:
            parts.append(_FILLER.strip())

        parts.append("- Northwind Support")
        reply = " ".join(parts)
        if max_words and len(reply.split()) > max_words and len(policies) > 1 and cite:
            parts = [policies[0] if p == " ".join(policies) else p for p in parts]
            reply = " ".join(parts)
        return reply


_NEXT_STEP_KNOWN_ORDER = "Next step: reply to this message if anything is still unclear."
_INVENTED_COMMITMENT = ("I've escalated this to our senior team and you'll hear back within "
                        "24 hours.")
_RUDE = ("calm down", "you should have", "obviously", "as i said", "your fault",
         "not our problem", "read the policy")
_EMPATHY_WORDS = ("sorry", "apolog", "understand", "frustrat")


def triage_thread(ticket: str, history: list[dict[str, str]], *, lite: bool = False
                  ) -> dict[str, Any]:
    """Offline triage for a conversation: the category can come from earlier customer
    messages, the tone and urgency come from the latest one."""
    latest = keyword_triage(ticket, lite=lite)
    earlier = " ".join(m["text"] for m in history if m.get("role") == "customer")
    if earlier and latest["category"] == "other":
        latest["category"] = keyword_triage(earlier + " " + ticket, lite=lite)["category"]
        latest["escalate"] = latest["category"] == "safety_legal" or latest["escalate"]
    return latest


def fake_judge(ctx: dict[str, Any]) -> dict[str, Any]:
    """Offline judge built from the deterministic rules. Lets the judge plumbing and the
    calibration maths run in CI; it is not evidence that an LLM judge works."""
    from .kb import tokens
    from .rules import commitment_violations, issue_kind

    reply = ctx.get("reply", "")
    policies = ctx.get("policies") or []
    customer = " ".join([ctx.get("ticket", "")] + [m["text"] for m in ctx.get("history") or []
                                                   if m.get("role") == "customer"])
    issues = commitment_violations(reply, policies, customer)
    kinds = {issue_kind(i) for i in issues}
    low = reply.lower()
    upset = bool(re.search(r"frustrat|annoy|angry|furious|unacceptable|ridiculous|upset|"
                           r"disappoint|terrible|worst", customer, re.I))
    overlap = set(tokens(reply)) & set(tokens(customer + " " + " ".join(policies)))
    verdict = {
        "grounded": not ({"invented timeline", "invented menu path"} & kinds),
        "answers_question": len(overlap) >= 3,
        "tone_ok": not any(r in low for r in _RUDE)
                   and (not upset or any(w in low for w in _EMPATHY_WORDS)),
        "no_invented_commitments": "invented commitment" not in kinds,
    }
    failed = [k for k, v in verdict.items() if not v]
    return {**verdict, "reason": "ok" if not failed else "fails " + ", ".join(failed)}


REQUEST_TIMEOUT_S = 60.0
MAX_RETRIES = 4  # the SDKs retry 429s, 5xx and connection errors with backoff on their own


class AnthropicModel:
    def __init__(self, model: str, client: Any = None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - needs optional extra
                raise RuntimeError('Install the live extras: pip install -e ".[live]"') from exc
            client = anthropic.Anthropic(max_retries=MAX_RETRIES,
                                         timeout=REQUEST_TIMEOUT_S)  # reads ANTHROPIC_API_KEY
        self._client = client
        self._model = model
        self.name = f"anthropic:{model}"

    def complete(self, task: str, system: str, user: str, *,
                 context: dict[str, Any] | None = None) -> str:
        started = time.perf_counter()
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        _record_from(getattr(resp, "usage", None), "input_tokens", "output_tokens",
                     started, system, user, text)
        return text


def _record_from(usage: Any, in_attr: str, out_attr: str, started: float, system: str,
                 user: str, text: str) -> None:
    latency = int((time.perf_counter() - started) * 1000)
    tin = getattr(usage, in_attr, None) if usage is not None else None
    tout = getattr(usage, out_attr, None) if usage is not None else None
    if isinstance(tin, int) and isinstance(tout, int):
        record_usage(tin, tout, latency)
    else:  # the server did not report usage
        record_usage(estimate_tokens(system, user), estimate_tokens(text), latency,
                     estimated=True)


OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"


class OpenAIModel:
    """OpenAI, or any server that speaks the same API: Ollama, LM Studio, vLLM, llama.cpp.

    `json_mode` asks the server for a JSON object on triage calls. Ollama supports this
    (`response_format={"type": "json_object"}`), which makes small local models far more
    reliable at returning parseable triage.
    """

    def __init__(self, model: str, client: Any = None, *, base_url: str | None = None,
                 api_key: str | None = None, provider: str = "openai",
                 json_mode: bool = False) -> None:
        if client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - needs optional extra
                raise RuntimeError('Install the live extras: pip install -e ".[live]"') from exc
            client = openai.OpenAI(base_url=base_url, api_key=api_key,
                                   max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT_S)
        self._client = client
        self._model = model
        self._json_mode = json_mode
        self.name = f"{provider}:{model}"

    def complete(self, task: str, system: str, user: str, *,
                 context: dict[str, Any] | None = None) -> str:
        kwargs: dict[str, Any] = {}
        if self._json_mode and task in ("triage", "judge"):
            kwargs["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **kwargs,
        )
        text = resp.choices[0].message.content or ""
        _record_from(getattr(resp, "usage", None), "prompt_tokens", "completion_tokens",
                     started, system, user, text)
        return text


class CachedModel:
    """Remember every response on disk so a rerun is free and exactly reproducible.

    The key covers the model, task, prompts and a repeat index, so `--repeats 3` makes three
    separate live calls per prompt while a second run of the same command makes none.
    """

    def __init__(self, inner: ChatModel, cache_dir: str | Path, repeat: int = 0) -> None:
        self._inner = inner
        self._dir = Path(cache_dir)
        self._repeat = repeat
        self.name = inner.name
        self.hits = 0
        self.misses = 0

    def _path(self, task: str, system: str, user: str) -> Path:
        raw = "\x1f".join([self.name, task, system, user, str(self._repeat)])
        return self._dir / f"{hashlib.sha256(raw.encode()).hexdigest()}.json"

    def complete(self, task: str, system: str, user: str, *,
                 context: dict[str, Any] | None = None) -> str:
        """On a hit, the usage recorded with the original call is replayed (marked cached), so
        latency budgets measure the model, not the cache. Entries written before usage was
        recorded get an estimate and no latency."""
        path = self._path(task, system, user)
        if path.exists():
            self.hits += 1
            entry = json.loads(path.read_text(encoding="utf-8"))
            u = entry.get("usage")
            if u:
                record_usage(u["input_tokens"], u["output_tokens"], u.get("latency_ms"),
                             cached=True, estimated=bool(u.get("estimated")))
            else:
                record_usage(estimate_tokens(system, user), estimate_tokens(entry["text"]),
                             None, cached=True, estimated=True)
            return entry["text"]
        pop_usage()
        text = self._inner.complete(task, system, user, context=context)
        usage = pop_usage()
        if usage:
            record_usage(usage["input_tokens"], usage["output_tokens"], usage["latency_ms"],
                         estimated=usage["estimated"])
        self.misses += 1
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"model": self.name, "task": task, "text": text,
                                   "usage": usage}), encoding="utf-8")
        tmp.replace(path)  # atomic, so an interrupted run never leaves a half-written entry
        return text


LIVE_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
LIVE_PROVIDERS = ("anthropic", "openai", "ollama", "openai_compat")


def missing_key(spec: str) -> str | None:
    """Name of the API-key env var a live model spec needs but does not have, else None."""
    var = LIVE_KEYS.get(spec.partition(":")[0])
    return var if var and not os.environ.get(var) else None


def get_model(spec: str = "fake", *, cache_dir: str | Path | None = None, repeat: int = 0,
              client: Any = None) -> ChatModel:
    provider, _, rest = spec.partition(":")
    if provider == "fake":
        return FakeModel(rest or "default")
    live: ChatModel
    if provider == "anthropic" and rest:
        live = AnthropicModel(rest, client)
    elif provider == "openai" and rest:
        live = OpenAIModel(rest, client)
    elif provider == "ollama" and rest:          # free, local, no key
        live = OpenAIModel(rest, client, provider="ollama", json_mode=True, api_key="ollama",
                           base_url=os.environ.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_URL))
    elif provider == "openai_compat" and rest:   # any other OpenAI-compatible server
        base_url = os.environ.get("CX_OPENAI_COMPAT_BASE_URL")
        if not base_url and client is None:
            raise ValueError("openai_compat:<model> needs CX_OPENAI_COMPAT_BASE_URL")
        live = OpenAIModel(rest, client, provider="openai_compat", base_url=base_url,
                           api_key=os.environ.get("CX_OPENAI_COMPAT_API_KEY", "none"))
    else:
        raise ValueError(f"Bad model spec {spec!r}. Try 'fake', 'fake:lite', 'ollama:<model>', "
                         "'openai_compat:<model>', 'anthropic:<model>' or 'openai:<model>'.")
    return CachedModel(live, cache_dir, repeat) if cache_dir else live
