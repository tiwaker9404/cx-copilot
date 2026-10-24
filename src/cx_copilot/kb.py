"""A tiny policy knowledge base for the fictional shop "Northwind Goods".

Retrieval is deliberately simple (keyword hits + a category boost) so that it is
deterministic and easy to reason about in tests. Swap `retrieve` for a vector
store later; the rest of the graph only depends on its signature.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    id: str
    category: str
    keywords: tuple[str, ...]
    text: str


POLICIES: tuple[Policy, ...] = (
    Policy(
        "refund_window",
        "returns",
        ("refund", "return", "money back", "exchange", "send back"),
        "Refunds are available within 30 days of delivery for unused items in original "
        "packaging. Approved refunds go back to the original payment method in "
        "5-7 business days.",
    ),
    Policy(
        "warranty",
        "returns",
        ("broken", "defective", "damaged", "cracked", "warranty", "stopped working",
         "doesn't work"),
        "Items are covered by a 12-month warranty against defects. We send a free "
        "replacement once you share a photo of the defect.",
    ),
    Policy(
        "shipping_times",
        "shipping",
        ("shipping", "delivery", "arrive", "tracking", "how long"),
        "Standard shipping takes 3-5 business days and express takes 1-2 business days. "
        "A tracking email is sent when your order ships.",
    ),
    Policy(
        "lost_package",
        "shipping",
        ("lost", "missing", "never arrived", "stuck", "hasn't moved", "hasn't updated"),
        "If tracking shows no movement for 7 days, we reissue the order or refund it in "
        "full. Please contact us within 30 days of the ship date.",
    ),
    Policy(
        "delivered_not_received",
        "shipping",
        ("delivered", "never got", "didn't receive", "not received"),
        "If tracking says delivered but nothing arrived, check with neighbors and "
        "household members for 24 hours, then contact us and we will open a carrier "
        "investigation.",
    ),
    Policy(
        "billing_duplicate",
        "billing",
        ("charged twice", "double charge", "duplicate", "twice", "pending"),
        "Confirmed duplicate charges are reversed within 3-5 business days. A pending "
        "authorization drops off automatically within 5 days.",
    ),
    Policy(
        "billing_subscription",
        "billing",
        ("cancel", "subscription", "renew", "membership"),
        "You can cancel any time under Settings > Billing. Access continues until the "
        "end of the paid period.",
    ),
    Policy(
        "password_reset",
        "account",
        ("password", "locked out", "log in", "login", "sign in"),
        "Use Forgot password on the sign-in page. The reset link expires after "
        "60 minutes.",
    ),
    Policy(
        "email_change",
        "account",
        ("email",),
        "Changing the email on an account requires a verification code sent to the "
        "current email address.",
    ),
    Policy(
        "app_crash",
        "technical",
        ("crash", "freez", "won't open", "not loading", "app"),
        "Update to the latest app version and restart. If the problem continues, send "
        "your device model and OS version so engineering can investigate.",
    ),
    Policy(
        "site_error",
        "technical",
        ("website", "check out", "checkout", "error", "browser"),
        "If the website shows an error, clear your browser cache and try again in a "
        "private window. If it persists, send us the page link and a screenshot.",
    ),
)


def retrieve(ticket: str, category: str, k: int = 2) -> list[Policy]:
    """Return up to `k` policies. A policy needs at least one keyword hit to qualify."""
    text = ticket.lower()
    scored: list[tuple[int, str, Policy]] = []
    for policy in POLICIES:
        hits = sum(1 for kw in policy.keywords if kw in text)
        if hits == 0:
            continue
        score = hits * 2 + (1 if policy.category == category else 0)
        scored.append((-score, policy.id, policy))
    scored.sort()
    return [p for _, _, p in scored[:k]]
