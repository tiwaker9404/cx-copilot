import pytest
from fastapi.testclient import TestClient

from cx_copilot.api import create_app
from cx_copilot.config import Settings
from cx_copilot.graph import HOLDING_REPLY, build_graph, run_ticket
from cx_copilot.harvest import candidates
from cx_copilot.models import FakeModel
from cx_copilot.service import Conflict, TicketService
from cx_copilot.store import Store

THREAD = [{"role": "customer", "text": "I was charged twice for my order."},
          {"role": "agent", "text": "Sorry about that. Please reply with your order number."}]


# ------------------------------------------------------------------ graph
def test_third_contact_goes_to_a_human():
    g = build_graph(FakeModel())
    two = THREAD + [{"role": "customer", "text": "Still nothing."},
                    {"role": "agent", "text": "Duplicates are reversed in 3-5 business days."}]
    out = run_ticket(g, "Order #2290, both charges are still there.", history=two)
    assert out["escalated"] and out["escalation_reason"] == "repeat_contact"
    assert out["reply"] == HOLDING_REPLY and "Earlier messages" in out["handoff"]


def test_second_contact_uses_the_thread():
    out = run_ticket(build_graph(FakeModel(), "reply_v4_grounded"),
                     "Order #2290, both charges are still there.", history=THREAD)
    assert not out["escalated"] and out["triage"]["category"] == "billing"
    assert out["reply"].startswith("Thanks for following up")
    assert "order number" not in out["reply"].lower()


def test_safety_words_earlier_in_the_thread_still_escalate():
    hist = [{"role": "customer", "text": "My charger started smoking."},
            {"role": "agent", "text": "Thanks, which model is it?"}]
    out = run_ticket(build_graph(FakeModel()), "It's the 20W one.", history=hist)
    assert out["escalated"] and out["escalation_reason"] == "safety_rule"


def test_review_needs_a_checkpointer():
    with pytest.raises(ValueError, match="checkpointer"):
        build_graph(FakeModel(), review="risky")


# ---------------------------------------------------------------- service
@pytest.fixture
def review_svc(tmp_path):
    s = Settings(db_path=str(tmp_path / "cx.db"), review_mode="risky",
                 checkpoint_path=str(tmp_path / "ck.db"))
    return s, TicketService(s)


def test_risky_reply_is_held_and_calm_one_is_not(review_svc):
    _, svc = review_svc
    held = svc.handle("I was charged twice and I'm really frustrated", "r1")
    assert held["status"] == "pending_review" and "frustrated" in held["review_reason"]
    assert svc.handle("How long does express shipping take?", "r2")["status"] == "ok"
    assert [r["ticket_id"] for r in svc.pending_reviews(10)] == ["r1"]


def test_paused_review_survives_a_restart_and_resumes_once(review_svc):
    settings, svc = review_svc
    svc.handle("I was charged twice and I'm really frustrated", "r1")
    restarted = TicketService(settings)
    out = restarted.review("r1", "edit", "Sorry! Confirmed duplicates are reversed within "
                                         "3-5 business days.", note="shorter", reviewer="ash")
    assert out["status"] == "ok" and out["reply"].startswith("Sorry!")
    assert out["review"]["action"] == "edit" and out["review"]["draft"]
    with pytest.raises(Conflict):
        restarted.review("r1", "approve")
    assert restarted.store.stats()["reviews"] == {"edit": 1}


def test_approve_keeps_the_draft_and_reject_escalates(review_svc):
    _, svc = review_svc
    draft = svc.handle("I'm annoyed, I can't log in, I forgot my password", "a")["reply"]
    assert svc.review("a", "approve")["reply"] == draft
    svc.handle("I'm annoyed, the app keeps crashing", "b")
    out = svc.review("b", "reject", note="needs engineering")
    assert out["escalated"] and out["escalation_reason"] == "review_rejected"
    assert out["reply"] == HOLDING_REPLY
    assert out in svc.escalations(10) or any(e["ticket_id"] == "b" for e in svc.escalations(10))


def test_bad_review_requests(review_svc):
    _, svc = review_svc
    svc.handle("How long does express shipping take?", "calm")
    with pytest.raises(Conflict):
        svc.review("calm", "approve")
    svc.handle("I'm really frustrated, my order is late", "late")
    with pytest.raises(ValueError, match="edit needs"):
        svc.review("late", "edit", "  ")


def test_history_is_redacted_and_part_of_idempotency(tmp_path):
    svc = TicketService(Settings(db_path=":memory:"), store=Store(":memory:"))
    hist = [{"role": "customer", "text": "My email is jane.doe@example.com"},
            {"role": "agent", "text": "Thanks."}]
    out = svc.handle("I never got the reset email", "h1", hist)
    assert "jane.doe" not in str(out) and "email" in out["pii_removed"]
    assert svc.handle("I never got the reset email", "h1", hist)["replayed"]
    with pytest.raises(Conflict):
        svc.handle("I never got the reset email", "h1", [])


def test_harvest_turns_edits_and_low_ratings_into_candidates(review_svc):
    _, svc = review_svc
    svc.handle("I was charged twice and I'm really frustrated", "e1")
    svc.review("e1", "edit", "Sorry! Duplicates are reversed within 3-5 business days.", "tone")
    svc.handle("How long does express shipping take?", "f1")
    svc.feedback("f1", 0.0, "too vague")
    rows = {r["id"]: r for r in candidates(svc.store)}
    assert rows["h_e1"]["reference_reply"].startswith("Sorry!")
    assert rows["h_e1"]["needs_labels"] and rows["h_e1"]["gold"]["category"] == "billing"
    assert rows["h_f1"]["sources"] == ["low_feedback"] and rows["h_f1"]["feedback_score"] == 0.0
    assert candidates(svc.store, existing={"h_e1", "h_f1"}) == []


# -------------------------------------------------------------------- API
def test_api_review_flow(tmp_path):
    s = Settings(db_path=str(tmp_path / "cx.db"), review_mode="all",
                 checkpoint_path=str(tmp_path / "ck.db"))
    client = TestClient(create_app(s))
    r = client.post("/v1/tickets", json={"text": "How do I cancel my subscription?",
                                         "ticket_id": "x1"})
    assert r.status_code == 200 and r.json()["status"] == "pending_review"
    assert [t["ticket_id"] for t in client.get("/v1/reviews").json()] == ["x1"]
    assert client.post("/v1/tickets/x1/review", json={"action": "edit"}).status_code == 422
    assert client.post("/v1/tickets/nope/review", json={"action": "approve"}).status_code == 404
    ok = client.post("/v1/tickets/x1/review", json={"action": "approve", "reviewer": "ash"})
    assert ok.status_code == 200 and ok.json()["review"]["reviewer"] == "ash"
    assert client.post("/v1/tickets/x1/review", json={"action": "approve"}).status_code == 409
    assert client.get("/v1/stats").json()["reviews"] == {"approve": 1}


def test_api_accepts_threads_and_limits_them(tmp_path):
    s = Settings(db_path=str(tmp_path / "cx.db"), max_history=2)
    client = TestClient(create_app(s))
    r = client.post("/v1/tickets", json={"text": "Order #2290, still charged twice.",
                                         "history": THREAD})
    assert r.status_code == 200 and r.json()["history"][0]["role"] == "customer"
    too_many = client.post("/v1/tickets", json={"text": "x", "history": THREAD * 2})
    assert too_many.status_code == 413
    bad_role = client.post("/v1/tickets", json={"text": "x",
                                                "history": [{"role": "bot", "text": "y"}]})
    assert bad_role.status_code == 422


def test_review_mode_is_validated(monkeypatch):
    monkeypatch.setenv("CX_REVIEW_MODE", "sometimes")
    with pytest.raises(ValueError, match="CX_REVIEW_MODE"):
        Settings.from_env()
