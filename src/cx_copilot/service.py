"""The ticket-handling service: everything between an HTTP request and the LangGraph agent.

Guarantees, each covered by a test:
  * PII is removed before the model, the trace, the database or the logs see the ticket.
  * The same ticket id is never processed twice (safe client retries).
  * If the model is down or misbehaves, the ticket is escalated to a human, never dropped.
  * A LangSmith outage never fails a request; tracing is best-effort.
  * Feedback is attached to the exact trace, without upgrading it to paid extended retention.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from typing import Any

from langsmith import traceable, tracing_context

from . import kb
from .config import Settings
from .graph import HOLDING_REPLY, build_graph
from .models import ChatModel, get_model
from .pii import redact
from .store import Store, now

log = logging.getLogger("cx_copilot")


class Busy(Exception):
    """No free model slot within the queue timeout."""


class Conflict(Exception):
    """A ticket id was reused with different text."""


class NotFound(Exception):
    pass


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TicketService:
    def __init__(self, settings: Settings, *, model: ChatModel | None = None,
                 store: Store | None = None, langsmith_client: Any = None) -> None:
        self.settings = settings
        self.model = model or get_model(settings.model)
        policies = kb.load_policies(settings.kb_path) if settings.kb_path else None
        self.graph = build_graph(self.model, settings.reply_prompt, policies,
                                 triage_prompt=settings.triage_prompt,
                                 escalation=settings.escalation)
        self.store = store or Store(settings.db_path)
        self._slots = threading.BoundedSemaphore(settings.max_concurrency)
        self._ls_client = langsmith_client
        # Wrap the bound method so the trace records (ticket, ticket_id), not `self`.
        self._traced = traceable(name="cx_ticket", run_type="chain")(self._invoke)

    # ---------------------------------------------------------------- tickets
    def handle(self, text: str, ticket_id: str | None = None) -> dict[str, Any]:
        ticket_id = ticket_id or f"tkt_{uuid.uuid4().hex[:16]}"
        clean, removed = redact(text.strip())
        digest = _hash(clean)

        existing = self.store.get(ticket_id)
        if existing is not None:
            if existing["text_hash"] != digest:
                raise Conflict(ticket_id)
            return {**existing["result"], "replayed": True}

        if not self._slots.acquire(timeout=self.settings.queue_timeout_s):
            raise Busy()
        try:
            result = self._run(ticket_id, clean, removed)
        finally:
            self._slots.release()

        if not self.store.save(ticket_id, digest, clean, result):  # lost a race with a retry
            stored = self.store.get(ticket_id)
            return {**stored["result"], "replayed": True}
        log.info("ticket handled", extra={"ticket_id": ticket_id, "status": result["status"],
                                          "escalated": result["escalated"],
                                          "latency_ms": result["latency_ms"]})
        return {**result, "replayed": False}

    def _run(self, ticket_id: str, clean: str, removed: list[str]) -> dict[str, Any]:
        run_id = uuid.uuid4()
        started = time.perf_counter()
        status, error = "ok", ""
        try:
            with tracing_context(enabled=self.settings.tracing, client=self._ls_client,
                                 project_name=self.settings.langsmith_project):
                state = self._traced(clean, ticket_id, langsmith_extra={
                    "run_id": run_id, "metadata": {"ticket_id": ticket_id,
                                                   "model": self.model.name,
                                                   "reply_prompt": self.settings.reply_prompt,
                                                   "triage_prompt": self.settings.triage_prompt,
                                                   "escalation": self.settings.escalation,
                                                   "pii_removed": removed}})
        except Exception as exc:  # model outage, timeout, bad output: hand it to a human
            status, error = "degraded", f"{type(exc).__name__}: {exc}"[:300]
            log.warning("model failure, escalating", extra={"ticket_id": ticket_id,
                                                            "error": error})
            state = {"triage": {"category": "other", "urgency": "high",
                                "sentiment": "neutral", "escalate": True},
                     "reply": HOLDING_REPLY, "escalated": True, "attempts": 0,
                     "policy_ids": [], "handoff": f"Automated handling failed ({error}). "
                                                  f"Customer wrote: {clean[:400]}"}
        triage = state["triage"]
        return {
            "ticket_id": ticket_id,
            "created_at": now(),
            "status": status,
            "category": triage["category"],
            "urgency": triage["urgency"],
            "sentiment": triage["sentiment"],
            "escalated": bool(state.get("escalated")),
            "reply": state["reply"],
            "handoff": state.get("handoff", ""),
            "policy_ids": state.get("policy_ids", []),
            "attempts": state.get("attempts", 0),
            "pii_removed": removed,
            "model": self.model.name,
            "reply_prompt": self.settings.reply_prompt,
            "run_id": str(run_id),
            "traced": self.settings.tracing and status == "ok",
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    def _invoke(self, clean: str, ticket_id: str) -> dict[str, Any]:
        return self.graph.invoke({"ticket": clean}, config={
            "run_name": "cx_copilot_graph",
            "tags": ["cx-copilot", "service", f"model:{self.model.name}",
                     f"prompt:{self.settings.reply_prompt}"]})

    def get(self, ticket_id: str) -> dict[str, Any]:
        row = self.store.get(ticket_id)
        if row is None:
            raise NotFound(ticket_id)
        return row["result"]

    # --------------------------------------------------------------- feedback
    def feedback(self, ticket_id: str, score: float, comment: str = "") -> dict[str, Any]:
        result = self.get(ticket_id)
        clean_comment, _ = redact(comment)
        sent = False
        if result.get("traced"):
            try:
                client = self._ls_client
                if client is None:
                    from langsmith import Client
                    client = Client()
                client.create_feedback(
                    result["run_id"], key="agent_reply_quality", score=score,
                    comment=clean_comment or None,
                    extend_trace_retention=False,  # keep free-tier retention; no paid upgrade
                )
                sent = True
            except Exception as exc:  # stored locally either way
                log.warning("langsmith feedback failed", extra={"ticket_id": ticket_id,
                                                                "error": str(exc)[:200]})
        fid = self.store.add_feedback(ticket_id, score, clean_comment, sent)
        return {"feedback_id": fid, "ticket_id": ticket_id, "sent_to_langsmith": sent}

    # ------------------------------------------------------------------ misc
    def escalations(self, limit: int) -> list[dict[str, Any]]:
        return self.store.escalations(limit)

    def ready(self) -> dict[str, Any]:
        return {"store": self.store.ping(), "model": self.model.name,
                "tracing": self.settings.tracing, "free_slots": self._slots._value}
