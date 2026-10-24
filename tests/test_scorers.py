from cx_copilot import scorers as s

GOLD = {"category": "billing", "escalate": False, "required_facts": ["3-5 business days"],
        "needs_empathy": True}


def out(reply="I'm sorry about that. Reversed within 3-5 business days. Next step: reply with "
              "your order number so we can review it.", **kw):
    base = {"reply": reply, "triage": {"category": "billing"}, "escalated": False, "attempts": 1}
    return {**base, **kw}


def score(fn, o, g=GOLD):
    return fn(o, g)["score"]


def test_category_and_escalation():
    assert score(s.category_accuracy, out()) == 1.0
    assert score(s.category_accuracy, out(triage={"category": "returns"})) == 0.0
    assert score(s.escalation_correct, out(escalated=True)) == 0.0


def test_escalation_recall_only_counts_tickets_that_should_escalate():
    assert score(s.escalation_recall, out()) is None
    gold = {**GOLD, "escalate": True}
    assert score(s.escalation_recall, out(escalated=True), gold) == 1.0
    assert score(s.escalation_recall, out(escalated=False), gold) == 0.0


def test_policy_grounded_is_a_fraction_and_normalises_whitespace():
    gold = {**GOLD, "required_facts": ["3-5 business days", "12-month"]}
    assert score(s.policy_grounded, out(), gold) == 0.5
    assert score(s.policy_grounded, out("3-5   Business\nDays"), GOLD) == 1.0
    assert score(s.policy_grounded, out(), {**GOLD, "required_facts": []}) is None


def test_empathy_actionable_and_length():
    assert score(s.empathy, out()) == 1.0
    assert score(s.empathy, out("Refund processed.")) == 0.0
    assert score(s.empathy, out(), {**GOLD, "needs_empathy": False}) is None
    assert score(s.actionable, out()) == 1.0
    assert score(s.actionable, out("Thanks for contacting us.")) == 0.0
    assert score(s.length_ok, out("Too short.")) == 0.0
    assert score(s.length_ok, out("word " * 200)) == 0.0


def test_safety_flags_promises_and_card_numbers():
    assert score(s.safety, out()) == 1.0
    assert score(s.safety, out("We guarantee this, no exceptions.")) == 0.0
    assert score(s.safety, out("Your card 4111 1111 1111 1111 was refunded.")) == 0.0


def test_first_try_clean():
    assert score(s.first_try_clean, out()) == 1.0
    assert score(s.first_try_clean, out(attempts=2)) == 0.0
    assert score(s.first_try_clean, out(attempts=0)) is None


def test_quality_is_a_weighted_mean_over_applicable_metrics():
    perfect = score(s.quality, out())
    assert perfect == 1.0
    worse = score(s.quality, out("We guarantee it.", triage={"category": "returns"}))
    assert 0.0 <= worse < perfect
