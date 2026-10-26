from cx_copilot import scorers as s

GOLD = {"category": "billing", "escalate": False, "required_facts": ["3-5 business days"],
        "needs_empathy": True, "relevant_policies": ["billing_duplicate"]}
POLICY = "Confirmed duplicate charges are reversed within 3-5 business days."


def out(reply="I'm sorry about that. Reversed within 3-5 business days. Next step: reply with "
              "your order number so we can review it.", **kw):
    base = {"reply": reply, "triage": {"category": "billing"}, "escalated": False, "attempts": 1,
            "ticket": "I was charged twice.", "history": [], "policies": [POLICY],
            "policy_ids": ["billing_duplicate"], "first_issues": []}
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


def test_invented_commitments_in_the_sent_reply_are_caught():
    assert score(s.no_invented_commitments, out()) == 1.0
    bad = out("I'm sorry. I've issued a refund and you'll get it within 24 hours. Reply here.")
    assert score(s.no_invented_commitments, bad) == 0.0
    assert score(s.no_invented_commitments, out(escalated=True)) is None


def test_repeat_questions_only_count_when_the_customer_already_said_it():
    assert score(s.no_repeat_questions, out()) == 1.0
    assert score(s.no_repeat_questions, out(ticket="Charged twice on order #4471.")) == 0.0
    thread = [{"role": "customer", "text": "Order number is 88123."},
              {"role": "agent", "text": "Thanks."}]
    assert score(s.no_repeat_questions, out(history=thread)) == 0.0


def test_draft_commitment_free_reads_the_first_draft():
    assert score(s.draft_commitment_free, out()) == 1.0
    caught = out(first_issues=["invented timeline: '24 hours' is not in the policy notes"],
                 attempts=2)
    assert score(s.draft_commitment_free, caught) == 0.0
    format_only = out(first_issues=["too long (>150 words)"], attempts=2)
    assert score(s.draft_commitment_free, format_only) == 1.0


def test_followup_not_repeated():
    prev = out()["reply"]
    thread = [{"role": "customer", "text": "Charged twice."}, {"role": "agent", "text": prev}]
    assert score(s.followup_not_repeated, out()) is None
    assert score(s.followup_not_repeated, out(history=thread)) == 0.0
    fresh = out("Thanks for following up. Since both charges are still there, they drop off "
                "in 3-5 business days once confirmed. Next step: reply here if not.",
                history=thread)
    assert score(s.followup_not_repeated, fresh) == 1.0


def test_retrieval_recall():
    assert score(s.retrieval_recall, out()) == 1.0
    assert score(s.retrieval_recall, out(policy_ids=["refund_window"])) == 0.0
    assert score(s.retrieval_recall, out(), {**GOLD, "relevant_policies": []}) is None


def test_judge_metrics_are_none_without_a_judge():
    assert score(s.JUDGE_SCORERS[0], out()) is None
    verdict = {"grounded": True, "answers_question": False, "tone_ok": True,
               "no_invented_commitments": True}
    assert [score(f, out(judge=verdict)) for f in s.JUDGE_SCORERS] == [1.0, 0.0, 1.0, 1.0]
