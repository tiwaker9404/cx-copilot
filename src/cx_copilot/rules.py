"""Deterministic reply rules shared by the in-graph QA guardrail and the eval scorers.

Two families:
  * `safety_violations` / `quality_violations`: things no reply may ever contain.
  * `commitment_violations`: things a reply may only say if the policy notes say them.
    A support bot that promises "a refund within 24 hours" or "I've escalated this to a
    specialist" when no policy allows it creates a commitment the business never made.
    Every timeline, action and menu path in the reply must be backed by the policy notes,
    the customer's own words, or the one fixed holding-reply SLA.
"""

from __future__ import annotations

import re
from typing import Iterable

FORBIDDEN_PHRASES = (
    "guarantee",
    "100%",
    "no exceptions",
    "i promise",
    "we promise",
    "definitely",
    "our fault",
    "we are liable",
)
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")
PLACEHOLDER_RE = re.compile(r"\[[A-Za-z _]+\]|\{\{.*?\}\}")

MIN_WORDS = 15
MAX_WORDS = 150

# The holding reply's promise ("within 1 business day") is the one SLA the business owns.
ALLOWED_SLAS = ("1 business day",)


def word_count(text: str) -> int:
    return len(text.split())


def safety_violations(reply: str) -> list[str]:
    """Things a customer-facing reply must never contain."""
    lowered = reply.lower()
    found = [f"forbidden phrase: {p!r}" for p in FORBIDDEN_PHRASES if p in lowered]
    if CARD_RE.search(reply):
        found.append("echoes a card-number-like string")
    return found


def quality_violations(reply: str) -> list[str]:
    """Format and safety checks that need no context."""
    issues = safety_violations(reply)
    if not reply.strip():
        issues.append("empty reply")
    if word_count(reply) > MAX_WORDS:
        issues.append(f"too long (>{MAX_WORDS} words)")
    if PLACEHOLDER_RE.search(reply):
        issues.append("contains an unfilled placeholder")
    return issues


# ------------------------------------------------------------------ timelines
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14,
          "thirty": 30, "sixty": 60, "a": 1, "an": 1}
_N = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fourteen|thirty|sixty)"
DURATION_RE = re.compile(
    rf"\b{_N}(?:\s*(?:-|–|—|to)\s*{_N})?[\s-]*(?:business\s+|working\s+|calendar\s+)?"
    r"(minute|min|hour|hr|day|week|month|year)s?\b", re.IGNORECASE)
DEADLINE_RE = re.compile(
    r"\b(?:by|before)\s+(?:tomorrow|tonight|end of (?:the )?day|eod|the weekend|next week|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\bwithin the hour\b",
    re.IGNORECASE)
_UNIT = {"min": "minute", "hr": "hour"}


def _num(token: str) -> int:
    return int(token) if token.isdigit() else _WORDS[token.lower()]


def durations(text: str) -> set[tuple[int, int, str]]:
    """{(low, high, unit)}: "5-7 business days" -> (5, 7, "day"), "12-month" -> (12, 12, ...)."""
    out = set()
    for m in DURATION_RE.finditer(text):
        lo = _num(m.group(1))
        hi = _num(m.group(2)) if m.group(2) else lo
        unit = m.group(3).lower()
        out.add((lo, hi, _UNIT.get(unit, unit)))
    return out


def _allowed_duration(d: tuple[int, int, str], allowed: set[tuple[int, int, str]]) -> bool:
    if d in allowed:
        return True
    lo, hi, unit = d
    if lo != hi:
        return False
    values = {v for a, b, u in allowed if u == unit for v in (a, b)}
    # "up to 7 business days" is backed by "5-7 business days"; "15 days past the window" is
    # arithmetic on two backed figures (45 days since delivery - 30-day window)
    return lo in values or any(abs(a - b) == lo for a in values for b in values)


# -------------------------------------------------------------------- actions
# A sentence makes a commitment when it speaks for the business in the future or perfect tense.
COMMIT_RE = re.compile(
    r"\b(?:we(?:'ll| will| can| are going to|'re going to| have|'ve| would be happy to)|"
    r"i(?:'ll| will| can| have|'ve| am going to|'m going to| would be happy to)|"
    r"let me|you(?:'ll| will) (?:receive|get|see|hear|be)|"
    r"(?:our team|the team|a specialist|someone|a manager|an agent|they)(?:'ll| will)|"
    r"(?:has|have) been (?:issued|processed|approved|refunded|credited|expedited|upgraded|"
    r"escalated|forwarded|flagged|reported)|"
    r"is (?:on (?:its|the) way|being (?:processed|expedited|shipped|reviewed)))\b",
    re.IGNORECASE)
NEGATION_RE = re.compile(r"\b(?:not|cannot|can't|won't|unable|isn't|aren't|don't|no longer|"
                         r"unfortunately)\b|n't\b", re.IGNORECASE)

# label -> stems. A stem in the reply must also appear in a policy note, or it is invented.
ACTIONS: dict[str, tuple[str, ...]] = {
    "refund": ("refund", "money back", "reimburs"),
    "replacement": ("replacement", "replace it", "send you a new", "reship", "re-ship",
                    "resend", "send another"),
    "credit or discount": ("store credit", "account credit", "a credit", "credit your",
                           "discount", "coupon", "voucher", "promo code", "% off",
                           "free of charge", "on us", "compensat", "goodwill"),
    "fee waiver": ("waive",),
    "shipping upgrade": ("expedit", "upgrade", "priority shipping", "overnight", "rush"),
    "escalation or follow-up": ("escalat", "forward", "passed this", "passed your",
                                "specialist", "senior team", "manager", "supervisor",
                                "get back to you", "contact you", "reach out to you",
                                "follow up with you", "keep you updated", "look into this",
                                "investigate this", "flagged", "reported this"),
}

# ------------------------------------------------------------------ menu paths
_SEG = r"[A-Z][\w']*(?: [\w']+){0,2}"
PATH_RE = re.compile(rf"{_SEG}(?:\s*(?:>|→|->|»)\s*{_SEG})+")
_SEP_RE = re.compile(r"\s*(?:>|→|->|»)\s*")


def menu_paths(text: str) -> set[tuple[str, ...]]:
    """{("settings", "billing")}: first word of each segment, lower-cased."""
    out = set()
    for m in PATH_RE.finditer(text):
        segs = [[w.strip("'\"") for w in seg.split()] for seg in _SEP_RE.split(m.group(0))
                if seg.strip()]
        # "Go to Settings > Billing and press Cancel": the word next to each ">" counts
        out.add(tuple([segs[0][-1].lower()] + [s[0].lower() for s in segs[1:]]))
    return out


def _allowed_path(path: tuple[str, ...], allowed: set[tuple[str, ...]]) -> bool:
    """Any contiguous part of a real path is fine: "Orders > Find an order" is part of
    "Account > Orders > Find an order"."""
    n = len(path)
    return any(a[i:i + n] == path for a in allowed for i in range(len(a) - n + 1))


# --------------------------------------------------------- already-given info
ORDER_GIVEN_RE = re.compile(
    r"#\s?\d{3,}|\border\s*(?:number|no\.?|id)?\s*(?:is|:)?\s*#?[A-Z]{0,3}-?\d{3,}",
    re.IGNORECASE)
ASK_ORDER_RE = re.compile(
    r"\b(?:reply with|send(?: us| me)?|share|provide|what(?:'s| is)|confirm|include|"
    r"let (?:us|me) know|need|tell (?:us|me)|could you give)\b[^.?!]{0,40}"
    r"\border (?:number|id|#)", re.IGNORECASE)
EMAIL_GIVEN_RE = re.compile(r"\[EMAIL\]|[\w.+-]+@[\w-]+\.[\w.]+")
ASK_EMAIL_RE = re.compile(r"\b(?:reply with|send|share|provide|what(?:'s| is)|confirm|"
                          r"let (?:us|me) know)\b[^.?!]{0,30}\b(?:your )?email address\b",
                          re.IGNORECASE)

# Passing a compliment or a product idea to the team is courtesy, not a promise to the customer.
COURTESY_RE = re.compile(r"\b(?:feedback|suggestion|kind words|compliment|idea|comments?)\b",
                         re.IGNORECASE)

_STEM_RE = {label: [(s, re.compile(r"\b" + re.escape(s))) for s in stems]
            for label, stems in ACTIONS.items()}

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


_CLAUSE_RE = re.compile(r",?\s+but\s+|;|\s+however,?\s+|,\s+though\s+|,\s+so\s+",
                        re.IGNORECASE)


def _clauses(text: str) -> list[str]:
    """Sentences split at "but" / "however" / ";", so "X is not offered, but I can give you a
    discount" is checked as two clauses and the negation does not hide the offer."""
    return [c for s in _SENTENCE_RE.split(text) for c in _CLAUSE_RE.split(s) if c.strip()]


def commitment_violations(reply: str, policies: Iterable[str] = (), customer_text: str = "",
                          ) -> list[str]:
    """What the reply commits to or asks for that it should not.

    `policies` are the policy notes the agent was given; `customer_text` is everything the
    customer has written in this conversation (they may quote their own timelines, and
    anything they already gave must not be asked for again).
    """
    policy_text = " ".join(policies)
    policy_low = policy_text.lower()
    issues: list[str] = []

    allowed = durations(policy_text) | durations(customer_text) | durations(" ".join(ALLOWED_SLAS))
    for d in sorted(durations(reply)):
        if not _allowed_duration(d, allowed):
            lo, hi, unit = d
            span = f"{lo}-{hi}" if lo != hi else str(lo)
            issues.append(f"invented timeline: '{span} {unit}s' is not in the policy notes")
    for m in DEADLINE_RE.finditer(reply):
        if m.group(0).lower() not in customer_text.lower():
            issues.append(f"invented timeline: '{m.group(0)}' is not in the policy notes")

    seen: set[str] = set()
    for sentence in _clauses(reply):
        if not COMMIT_RE.search(sentence) or NEGATION_RE.search(sentence):
            continue
        low = sentence.lower()
        for label, stems in _STEM_RE.items():
            if label in seen:
                continue
            if label == "escalation or follow-up" and COURTESY_RE.search(sentence):
                continue
            hit = next((s for s, rx in stems if rx.search(low)), None)
            if hit and not any(rx.search(policy_low) for _, rx in stems):
                issues.append(f"invented commitment: offers {label} ('{hit}') that the "
                              "policy notes do not")
                seen.add(label)

    allowed_paths = menu_paths(policy_text)
    for path in sorted(menu_paths(reply)):
        if not _allowed_path(path, allowed_paths):
            issues.append(f"invented menu path: '{' > '.join(path)}' is not in the policy notes")

    if ORDER_GIVEN_RE.search(customer_text) and ASK_ORDER_RE.search(reply):
        issues.append("repeat question: asks for the order number the customer already gave")
    if EMAIL_GIVEN_RE.search(customer_text) and ASK_EMAIL_RE.search(reply):
        issues.append("repeat question: asks for the email the customer already gave")
    return issues


def issue_kind(issue: str) -> str:
    """Group a QA issue for reports: 'invented commitment', 'too long', ..."""
    for prefix in ("invented timeline", "invented commitment", "invented menu path",
                   "repeat question", "forbidden phrase", "too long", "empty reply",
                   "contains an unfilled placeholder", "echoes a card"):
        if issue.startswith(prefix):
            return prefix
    return "other"


INVENTED_KINDS = ("invented timeline", "invented commitment", "invented menu path")
