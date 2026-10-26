"""Strip personal data from tickets before the model, the trace, the database or the logs see it.

Data minimisation: nothing downstream needs a customer's card number or phone number to answer a
billing question, so it is replaced with a placeholder at the front door.
"""

from __future__ import annotations

import re

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("[CARD]", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    ("[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("[PHONE]", re.compile(r"(?<![\w#])(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}\b")),
    ("[SECRET]", re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{8,}\b")),
)


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def redact(text: str) -> tuple[str, list[str]]:
    """Return the redacted text and the kinds of data that were removed.

    Card-like numbers are only removed when they pass the Luhn check, so a long order or
    tracking number is not mangled.
    """
    found: list[str] = []
    for label, pattern in _RULES:
        if label == "[CARD]":
            hits = 0

            def card(m: re.Match[str]) -> str:
                nonlocal hits
                digits = re.sub(r"\D", "", m.group(0))
                if _luhn(digits):
                    hits += 1
                    return label
                return m.group(0)

            text, n = pattern.sub(card, text), hits
        else:
            text, n = pattern.subn(label, text)
        if n:
            found.append(label.strip("[]").lower())
    return text, found
