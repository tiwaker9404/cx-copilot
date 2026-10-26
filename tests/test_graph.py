import json

from cx_copilot.graph import HOLDING_REPLY, MAX_ATTEMPTS, build_graph, parse_triage, run_ticket
from cx_copilot.models import FakeModel

OK_TRIAGE = json.dumps({"category": "billing", "urgency": "normal", "sentiment": "neutral",
                        "escalate": False})
CLEAN = "I'm sorry for the trouble. Duplicate charges are reversed in 3-5 business days. " \
        "Next step: reply with your order number. - Northwind Support"


def test_parse_triage_accepts_fenced_json_and_rejects_junk():
    fenced = "```json\n" + OK_TRIAGE + "\n```"
    assert parse_triage(fenced)["category"] == "billing"
    assert parse_triage("sure! here is the answer") is None
    assert parse_triage(json.dumps({"category": "made_up"})) is None


def test_normal_ticket_gets_policy_grounded_reply():
    g = build_graph(FakeModel(), "reply_v4_grounded")
    out = run_ticket(g, "I was charged twice for order #4471")
    assert out["triage"]["category"] == "billing"
    assert "billing_duplicate" in out["policy_ids"]
    assert "3-5 business days" in out["reply"]
    assert out["attempts"] == 1 and not out["escalated"]


def test_asking_again_for_a_given_order_number_is_sent_back():
    """v1 asks for the order number even though the customer gave it; QA catches it."""
    out = run_ticket(build_graph(FakeModel(), "reply_v1_baseline"), "I was charged twice for order #4471")
    assert out["attempts"] == 2 and not out["escalated"]
    assert any(i.startswith("repeat question") for i in out["first_issues"])
    assert "order number" not in out["reply"].lower()


def test_dangerous_ticket_escalates_with_fixed_reply():
    out = run_ticket(build_graph(FakeModel()), "My charger is smoking and burned my desk")
    assert out["escalated"] and out["reply"] == HOLDING_REPLY and out["handoff"]
    assert "policy_ids" not in out  # never reached retrieval


def test_regex_safety_net_beats_a_model_that_says_no_escalation(scripted):
    m = scripted(OK_TRIAGE, [CLEAN])  # model claims: billing, no escalation
    out = run_ticket(build_graph(m), "I will sue you over this charge")
    assert out["escalated"] and out["triage"]["category"] == "safety_legal"
    assert m.reply_calls == 0


def test_unparseable_triage_fails_safe_to_a_human(scripted):
    m = scripted("I think this is a billing issue!", [CLEAN])
    out = run_ticket(build_graph(m), "hello?")
    assert out["escalated"] and out["reply"] == HOLDING_REPLY


def test_qa_sends_a_bad_draft_back_once_then_passes(scripted):
    m = scripted(OK_TRIAGE, ["We guarantee a refund, no exceptions!", CLEAN])
    out = run_ticket(build_graph(m), "I was charged twice")
    assert out["attempts"] == 2 and out["reply"] == CLEAN and not out["escalated"]


def test_qa_escalates_if_the_revision_is_still_bad(scripted):
    m = scripted(OK_TRIAGE, ["We guarantee it."])
    out = run_ticket(build_graph(m), "I was charged twice")
    assert out["attempts"] == MAX_ATTEMPTS
    assert out["escalated"] and out["reply"] == HOLDING_REPLY
    assert "QA failed" in out["handoff"]
