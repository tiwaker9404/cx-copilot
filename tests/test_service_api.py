"""The HTTP service: auth, validation, idempotency, PII, failure handling, feedback, limits."""

import json
import logging
import threading

import pytest
from fastapi.testclient import TestClient

from cx_copilot import kb
from cx_copilot.api import JsonFormatter, create_app
from cx_copilot.config import Settings
from cx_copilot.graph import HOLDING_REPLY
from cx_copilot.models import FakeModel
from cx_copilot.pii import redact
from cx_copilot.service import Busy, TicketService
from cx_copilot.store import Store

KEY = "k" * 20
CARD = "4111 1111 1111 1111"


class SpyModel(FakeModel):
    """FakeModel that records what it was sent and can be told to fail."""

    def __init__(self, fail=False):
        super().__init__()
        self.inputs, self.fail = [], fail

    def complete(self, task, system, user, *, context=None):
        self.inputs.append(user)
        if self.fail:
            raise TimeoutError("model did not answer")
        return super().complete(task, system, user, context=context)


from langsmith import Client as _LangSmithClient


class StubLangSmith(_LangSmithClient):
    """A real LangSmith client aimed at a closed port (trace uploads fail quietly, as they
    would during an outage), with create_feedback recorded instead of sent."""

    def __init__(self, fail=False):
        super().__init__(api_url="http://127.0.0.1:9", api_key="test")
        self.calls, self.fail = [], fail

    def create_feedback(self, run_id, **kw):
        if self.fail:
            raise ConnectionError("langsmith down")
        self.calls.append({"run_id": run_id, **kw})


def make(settings=None, model=None, ls=None, **kw):
    settings = settings or Settings(db_path=":memory:", **kw)
    svc = TicketService(settings, model=model or SpyModel(), store=Store(":memory:"),
                        langsmith_client=ls)
    return TestClient(create_app(settings, svc)), svc


def post(client, text, tid=None, key=None):
    headers = {"X-API-Key": key} if key else {}
    body = {"text": text} | ({"ticket_id": tid} if tid else {})
    return client.post("/v1/tickets", json=body, headers=headers)


# ------------------------------------------------------------ basics
def test_health_and_readiness():
    c, _ = make()
    assert c.get("/healthz").json()["status"] == "ok"
    ready = c.get("/readyz")
    assert ready.status_code == 200 and ready.json()["store"] is True


def test_ticket_round_trip_and_fetch():
    c, _ = make()
    r = post(c, "I was charged twice for order #4471", "t1")
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "billing" and "3-5 business days" in body["reply"]
    assert body["status"] == "ok" and not body["escalated"] and body["run_id"]
    assert c.get("/v1/tickets/t1").json()["reply"] == body["reply"]
    assert c.get("/v1/tickets/nope").status_code == 404


def test_risky_ticket_lands_in_the_escalation_queue():
    c, _ = make()
    assert post(c, "My charger is smoking and burned my desk").json()["escalated"]
    post(c, "How long does express shipping take?")
    queue = c.get("/v1/escalations").json()
    assert len(queue) == 1 and queue[0]["category"] == "safety_legal"


# ------------------------------------------------------------ input
@pytest.mark.parametrize("body", [{"text": ""}, {}, {"text": "hi", "ticket_id": "bad id!"},
                                  {"text": "hi", "ticket_id": "x" * 65}])
def test_invalid_input_is_rejected(body):
    c, _ = make()
    assert c.post("/v1/tickets", json=body).status_code == 422


def test_overlong_ticket_is_rejected():
    c, _ = make(max_ticket_chars=100)
    assert post(c, "a" * 101).status_code == 413


# ------------------------------------------------------------ idempotency
def test_resending_the_same_ticket_does_not_call_the_model_again():
    model = SpyModel()
    c, _ = make(model=model)
    first = post(c, "I was charged twice", "dup").json()
    calls = len(model.inputs)
    again = post(c, "I was charged twice", "dup").json()
    assert len(model.inputs) == calls and again["replayed"] and not first["replayed"]
    assert again["reply"] == first["reply"]


def test_reusing_a_ticket_id_with_different_text_is_a_conflict():
    c, _ = make()
    post(c, "I was charged twice", "dup")
    assert post(c, "Totally different ticket", "dup").status_code == 409


# ------------------------------------------------------------ PII
def test_pii_never_reaches_the_model_the_store_or_the_reply():
    model = SpyModel()
    c, svc = make(model=model)
    body = post(c, f"Charged twice on card {CARD}, email me at jo@example.com", "p1").json()
    assert set(body["pii_removed"]) == {"card", "email"}
    sent = " ".join(model.inputs)
    stored = json.dumps(svc.store.get("p1"))
    for secret in (CARD, "jo@example.com"):
        assert secret not in sent and secret not in stored and secret not in body["reply"]


def test_redaction_placeholders_do_not_trigger_unrelated_policies():
    found = kb.retrieve("charged twice, contact me at [EMAIL]", "account")
    assert "email_change" not in [p.id for p in found]


def test_mentioning_email_is_not_the_same_as_asking_to_change_it():
    assert "email_change" not in [p.id for p in kb.retrieve("Email me when it ships", "shipping")]
    asks = kb.retrieve("I need to change the email on my account", "account")
    assert [p.id for p in asks] == ["email_change"]


def test_redact_keeps_order_numbers_and_luhn_checks_cards():
    text, kinds = redact(f"order 1234567890123456 card {CARD} call (415) 555-0134 #4471")
    assert "1234567890123456" in text and "#4471" in text and CARD not in text
    assert kinds == ["card", "phone"]


# ------------------------------------------------------------ failures
def test_model_outage_escalates_to_a_human_instead_of_failing():
    c, _ = make(model=SpyModel(fail=True))
    r = post(c, "Where is my order?", "down1")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "degraded" and body["escalated"]
    assert body["reply"] == HOLDING_REPLY and "TimeoutError" in body["handoff"]
    assert c.get("/v1/stats").json()["degraded"] == 1


def test_no_free_model_slot_returns_503_with_retry_after():
    settings = Settings(db_path=":memory:", max_concurrency=1, queue_timeout_s=1)
    c, svc = make(settings)
    svc._slots.acquire()  # simulate a long-running request holding the only slot
    try:
        r = post(c, "hello")
        assert r.status_code == 503 and r.headers["retry-after"] == "5"
    finally:
        svc._slots.release()


def test_concurrent_duplicate_requests_store_one_result():
    c, svc = make()
    results = []

    def go():
        results.append(post(c, "I was charged twice", "race").json())

    threads = [threading.Thread(target=go) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({r["run_id"] for r in results}) == 1
    assert svc.store.stats()["tickets"] == 1


# ------------------------------------------------------------ auth + limits
def test_prod_requires_a_valid_api_key():
    settings = Settings(env="prod", model="ollama:x", api_keys=(KEY,), db_path=":memory:")
    c, _ = make(settings, model=SpyModel())
    assert post(c, "hi").status_code == 401
    assert post(c, "hi", key="wrong-key-wrong-key").status_code == 401
    assert post(c, "hi", key=KEY).status_code == 200
    assert c.get("/healthz").status_code == 200  # probes stay open


def test_rate_limit_returns_429():
    c, _ = make(rate_limit_per_min=2)
    assert [post(c, f"ticket {i}").status_code for i in range(3)] == [200, 200, 429]


def test_prod_settings_are_validated_at_startup(monkeypatch):
    monkeypatch.setenv("CX_ENV", "prod")
    monkeypatch.delenv("CX_API_KEYS", raising=False)
    with pytest.raises(ValueError, match="CX_API_KEYS"):
        Settings.from_env()
    monkeypatch.setenv("CX_API_KEYS", "short")
    with pytest.raises(ValueError, match="16 characters"):
        Settings.from_env()
    monkeypatch.setenv("CX_API_KEYS", KEY)
    with pytest.raises(ValueError, match="fake model"):
        Settings.from_env()
    monkeypatch.setenv("CX_MODEL", "ollama:llama3.2")
    assert Settings.from_env().auth_required
    monkeypatch.setenv("CX_MAX_CONCURRENCY", "zero")
    with pytest.raises(ValueError, match="integer"):
        Settings.from_env()


def test_docs_are_off_in_prod():
    settings = Settings(env="prod", model="ollama:x", api_keys=(KEY,), db_path=":memory:",
                        docs_enabled=False)
    c, _ = make(settings)
    assert c.get("/docs").status_code == 404


# ------------------------------------------------------------ feedback / LangSmith
def test_feedback_goes_to_the_trace_without_paid_retention():
    ls = StubLangSmith()
    settings = Settings(db_path=":memory:", tracing=True)
    c, svc = make(settings, ls=ls)
    body = post(c, "I was charged twice", "fb1").json()
    assert body["status"] == "ok" and body["traced"]
    out = c.post("/v1/tickets/fb1/feedback", json={"score": 0, "comment": "too long"}).json()
    assert out["sent_to_langsmith"] is True
    call = ls.calls[0]
    assert call["run_id"] == body["run_id"] and call["extend_trace_retention"] is False
    assert call["key"] == "agent_reply_quality" and call["score"] == 0


def test_feedback_is_kept_locally_when_tracing_is_off_or_langsmith_is_down():
    c, svc = make()
    post(c, "I was charged twice", "fb2")
    assert c.post("/v1/tickets/fb2/feedback", json={"score": 1}).json()["sent_to_langsmith"] is False
    ls = StubLangSmith(fail=True)
    c2, svc2 = make(Settings(db_path=":memory:", tracing=True), ls=ls)
    post(c2, "I was charged twice", "fb3")
    assert c2.post("/v1/tickets/fb3/feedback", json={"score": 1}).status_code == 200
    assert svc2.store.stats()["feedback"] == 1


def test_feedback_validation():
    c, _ = make()
    post(c, "hello", "fb4")
    assert c.post("/v1/tickets/fb4/feedback", json={"score": 2}).status_code == 422
    assert c.post("/v1/tickets/missing/feedback", json={"score": 1}).status_code == 404


def test_an_unreachable_langsmith_never_fails_a_request():
    from langsmith import Client
    dead = Client(api_url="http://127.0.0.1:9", api_key="x")
    c, _ = make(Settings(db_path=":memory:", tracing=True), ls=dead)
    r = post(c, "I was charged twice", "ls-down")
    assert r.status_code == 200 and r.json()["status"] == "ok"


# ------------------------------------------------------------ logging
def test_logs_are_json_and_never_contain_ticket_text(caplog):
    c, _ = make()
    caplog.set_level(logging.INFO, logger="cx_copilot")
    post(c, "My secret complaint about order 777", "log1")
    lines = [JsonFormatter().format(r) for r in caplog.records]
    assert lines and all(json.loads(line)["msg"] for line in lines)
    assert not any("secret complaint" in line for line in lines)


def test_request_id_is_echoed():
    c, _ = make()
    r = c.get("/healthz", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
