import pytest

from cx_copilot.rules import commitment_violations, durations, issue_kind, menu_paths

REFUND = ("Refunds are available within 30 days of delivery for unused items in original "
          "packaging. Approved refunds go back to the original payment method in 5-7 business days.")
SUB = "You can cancel any time under Settings > Billing."


def kinds(reply, policies=(REFUND,), customer=""):
    return [issue_kind(i) for i in commitment_violations(reply, policies, customer)]


def test_policy_backed_reply_is_clean():
    assert kinds("Sorry! Refunds are available within 30 days, and approved refunds go back "
                 "to your card in 5-7 business days. Next step: send it back unused.") == []


@pytest.mark.parametrize("reply", [
    "You'll have your money in 24-48 hours.",
    "Refunds take 10 business days.",
    "We'll sort this by tomorrow.",
])
def test_timelines_not_in_policy_are_invented(reply):
    assert "invented timeline" in kinds(reply)


def test_timelines_the_customer_gave_or_the_holding_sla_are_fine():
    assert kinds("I understand it has been 9 days.", customer="tracking stuck for 9 days") == []
    assert kinds("A specialist will reply within 1 business day.") == \
        ["invented commitment"]  # the SLA is fine, the escalation claim is not
    assert kinds("Our team will contact you soon.") == ["invented commitment"]
    assert kinds("Refunds arrive in up to 7 business days.") == []  # inside 5-7


@pytest.mark.parametrize("reply,label", [
    ("I've expedited your order.", "shipping upgrade"),
    ("We'll send you a 10% off coupon.", "credit or discount"),
    ("I have escalated this to a specialist.", "escalation or follow-up"),
    ("Let me look into this for you.", "escalation or follow-up"),
    ("I'll waive the fee this time.", "fee waiver"),
    ("Price matching is not offered, but I can give you a discount.", "credit or discount"),
])
def test_offers_the_policy_does_not_make_are_invented(reply, label):
    issues = commitment_violations(reply, [REFUND])
    assert any(label in i for i in issues), issues


def test_negated_and_policy_backed_offers_are_fine():
    assert kinds("We can't offer a discount on this order.") == []
    assert kinds("We'll refund it once it arrives back unused.") == []  # refund is in policy
    assert kinds("Thanks for the kind words, I'm glad it fits.") == []


def test_menu_paths_must_come_from_the_policy():
    assert kinds("Go to Settings > Billing and press Cancel.", [SUB]) == []
    assert kinds("Open Account > Subscriptions > Manage.", [SUB]) == ["invented menu path"]
    assert menu_paths("Settings > Privacy > Delete account") == {("settings", "privacy", "delete")}


def test_asking_again_for_given_details():
    ask = "Next step: reply with your order number."
    assert kinds(ask, customer="I was charged twice.") == []
    assert kinds(ask, customer="Charged twice on order #4471") == ["repeat question"]
    assert kinds("Could you confirm your email address?", customer="My email is [EMAIL]") == \
        ["repeat question"]


def test_duration_parsing():
    assert durations("5-7 business days, a 12-month warranty, five to seven days, 60 minutes") \
        == {(5, 7, "day"), (12, 12, "month"), (60, 60, "minute")}


# false alarms found by auditing the real qwen2.5:7b run (v0.3)
def test_audit_arithmetic_on_backed_figures_is_not_invented():
    reply = "It's been 45 days, which is 15 days past our 30 days window."
    assert kinds(reply, customer="It's been 45 days since delivery") == []


def test_audit_quoted_and_partial_menu_paths():
    pol = ["Add one under Account > Orders > Find an order, using the order number and email."]
    assert kinds("Go to 'Account > Orders' to see it.", pol) == []
    assert kinds("Open Orders > Find an order.", pol) == []
    assert kinds("Open Orders > Returns.", pol) == ["invented menu path"]


def test_audit_passing_on_a_compliment_is_courtesy():
    assert kinds("Thanks! I'll forward your suggestion to our product team.", []) == []
    assert kinds("I'll forward your case to billing.", []) == ["invented commitment"]
