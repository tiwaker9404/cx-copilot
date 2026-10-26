"""The ticket-handling service: everything between an HTTP request and the LangGraph agent.

Guarantees, each covered by a test:
  * PII is removed before the model, the trace, the database or the logs see the ticket
    (and every earlier message in the thread).
  * The same ticket id is never processed twice (safe client retries).
  * If the model is down or misbehaves, the ticket is escalated to a human, never dropped.
  * A LangSmith outage never fails a request; tracing is best-effort.
  * Feedback is attached to the exact trace, without upgrading it to paid extended retention.
  * With CX_REVIEW_MODE=risky|all, a reply that needs a person is held as `pending_review`
    and never returned as sendable. The paused run is checkpointed in SQLite, so a restart
    does not lose it, and only one reviewer can resume it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from langgraph.types import Command
from langsmith import traceable, tracing_context

from . import kb
from .config import Settings
from .graph import HOLDING_REPLY, build_graph, run_ticket
from .models import ChatModel, get_model
from .pii import redact
from .rules import quality_violations
from .store import Store, now

log = logging.getLogger("cx_copilot")

REVIEW_ACTIONS = ("approve", "edit", "reject")


class Busy(Exception):
    """No free model slot within the queue timeout."""


class Conflict(Exception):
    """A ticket id was reused with different text, or a review is not possible."""


class NotFound(Exception):
    pass


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TicketService:
    def __init__(self, settings: Settings, *, model: ChatModel | None = None,
                 store: Store | None = None, langsmith_client: Any = None,
                 embedder: kb.Embedder | None = None) -> None:
        self.settings = settings
        self.model = model or get_model(settings.model)
        policies = kb.load_policies(settings.kb_path) if settings.kb_path else None
        embedder = embedder or kb.get_embedder(settings.embed_model)
        self.retriever = kb.Retriever(policies, settings.retrieval, embedder)
        self.checkpointer = None
        if settings.review_mode != "off":
            from langgraph.checkpoint.sqlite import SqliteSaver

            if settings.checkpoint_path != ":memory:":
                Path(settings.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
            self.checkpointer = SqliteSaver(conn)
        self.graph = build_graph(self.model, settings.reply_prompt,
                                 triage_prompt=settings.triage_prompt,
                                 escalation=settings.escalation, retriever=self.retriever,
                                 review=settings.review_mode, checkpointer=self.checkpointer)
        self.store = store or Store(settings.db_path)
        self._slots = threading.BoundedSemaphore(settings.max_concurrency)
        self._ls_client = langsmith_client
        # Wrap bound methods so the trace records the ticket, not `self`.
        self._traced = traceable(name="cx_ticket", run_type="chain")(self._invoke)
        self._traced_resume = traceable(name="cx_review", run_type="chain")(self._resume)

    # ---------------------------------------------------------------- tickets
    def handle(self, text: str, ticket_id: str | None = None,
               history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        ticket_id = ticket_id or f"tkt_{uuid.uuid4().hex[:16]}"
        clean, removed = redact(text.strip())
        clean_history = []
        for m in history or []:
            t, kinds = redact(m["text"].strip())
            clean_history.append({"role": m["role"], "text": t})
            removed = sorted(set(removed) | set(kinds))
        digest = _hash(clean + "\x1f" + json.dumps(clean_history, sort_keys=True))

        existing = self.store.get(ticket_id)
        if existing is not None:
            if existing["text_hash"] != digest:
                raise Conflict(ticket_id)
            return {**existing["result"], "replayed": True}

        if not self._slots.acquire(timeout=self.settings.queue_timeout_s):
            raise Busy()
        try:
            result = self._run(ticket_id, clean, clean_history, removed)
        finally:
            self._slots.release()

        if not self.store.save(ticket_id, digest, clean, result):  # lost a race with a retry
            stored = self.store.get(ticket_id)
            return {**stored["result"], "replayed": True}
        log.info("ticket handled", extra={"ticket_id": ticket_id, "status": result["status"],
                                          "escalated": result["escalated"],
                                          "latency_ms": result["latency_ms"]})
        return {**result, "replayed": False}

    def _trace_meta(self, ticket_id: str, extra: dict[str, Any]) -> dict[str, Any]:
        return {"ticket_id": ticket_id, "model": self.model.name,
                "reply_prompt": self.settings.reply_prompt,
                "triage_prompt": self.settings.triage_prompt,
                "escalation": self.settings.escalation, "retrieval": self.retriever.name,
                "review_mode": self.settings.review_mode, **extra}

    def _run(self, ticket_id: str, clean: str, history: list[dict[str, str]],
             removed: list[str]) -> dict[str, Any]:
        run_id = uuid.uuid4()
        started = time.perf_counter()
        status, error = "ok", ""
        try:
            with tracing_context(enabled=self.settings.tracing, client=self._ls_client,
                                 project_name=self.settings.langsmith_project):
                state = self._traced(clean, ticket_id, history, langsmith_extra={
                    "run_id": run_id, "metadata": self._trace_meta(
                        ticket_id, {"pii_removed": removed, "turns": 1 + len(
                            [m for m in history if m["role"] == "customer"])})})
        except Exception as exc:  # model outage, timeout, bad output: hand it to a human
            status, error = "degraded", f"{type(exc).__name__}: {exc}"[:300]
            log.warning("model failure, escalating", extra={"ticket_id": ticket_id,
                                                            "error": error})
            state = {"triage": {"category": "other", "urgency": "high",
                                "sentiment": "neutral", "escalate": True},
                     "reply": HOLDING_REPLY, "escalated": True, "attempts": 0,
                     "escalation_reason": "model_failure",
                     "policy_ids": [], "handoff": f"Automated handling failed ({error}). "
                                                  f"Customer wrote: {clean[:400]}"}
        if "__interrupt__" in state:
            status = "pending_review"
        result = self._result(ticket_id, state, status, run_id, started)
        result.update(history=history, pii_removed=removed)
        return result

    def _result(self, ticket_id: str, state: dict[str, Any], status: str, run_id: Any,
                started: float, created_at: str | None = None) -> dict[str, Any]:
        triage = state["triage"]
        usage = state.get("usage") or {}
        return {
            "ticket_id": ticket_id,
            "created_at": created_at or now(),
            "status": status,
            "category": triage["category"],
            "urgency": triage["urgency"],
            "sentiment": triage["sentiment"],
            "escalated": bool(state.get("escalated")),
            "escalation_reason": state.get("escalation_reason", "")
            if state.get("escalated") else "",
            "reply": state["reply"],
            "handoff": state.get("handoff", ""),
            "policy_ids": state.get("policy_ids", []),
            "attempts": state.get("attempts", 0),
            "review_reason": state.get("review_reason", "") if status == "pending_review" else "",
            "review": state.get("review"),
            "tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            "model_calls": usage.get("calls", 0),
            "pii_removed": [],
            "model": self.model.name,
            "reply_prompt": self.settings.reply_prompt,
            "run_id": str(run_id),
            "traced": self.settings.tracing and status != "degraded",
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    def _config(self, ticket_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": ticket_id}} if self.checkpointer else {}

    def _invoke(self, clean: str, ticket_id: str, history: list[dict[str, str]]) -> dict[str, Any]:
        return run_ticket(self.graph, clean, history=history, model_name=self.model.name,
                          reply_prompt=self.settings.reply_prompt,
                          triage_prompt=self.settings.triage_prompt,
                          escalation=self.settings.escalation,
                          retrieval=self.retriever.name, ticket_id=ticket_id,
                          thread_id=ticket_id if self.checkpointer else None)

    def _resume(self, ticket_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        return self.graph.invoke(Command(resume=decision), config={
            **self._config(ticket_id), "run_name": "cx_copilot_review",
            "tags": ["cx-copilot", "service", "review", f"review:{decision['action']}"]})

    def get(self, ticket_id: str) -> dict[str, Any]:
        row = self.store.get(ticket_id)
        if row is None:
            raise NotFound(ticket_id)
        return row["result"]

    # ---------------------------------------------------------------- reviews
    def pending_reviews(self, limit: int) -> list[dict[str, Any]]:
        return self.store.pending_reviews(limit)

    def review(self, ticket_id: str, action: str, reply: str | None = None, note: str = "",
               reviewer: str = "anonymous") -> dict[str, Any]:
        """Approve, edit or reject a held reply, and resume the paused run."""
        result = self.get(ticket_id)
        if action not in REVIEW_ACTIONS:
            raise ValueError(f"action must be one of {', '.join(REVIEW_ACTIONS)}")
        if action == "edit" and not (reply or "").strip():
            raise ValueError("an edit needs the new reply text")
        if result["status"] != "pending_review" or self.checkpointer is None:
            raise Conflict(f"ticket {ticket_id} is not waiting for review")
        if not self.store.claim_review(ticket_id):
            raise Conflict(f"ticket {ticket_id} is already being reviewed")
        clean_note, _ = redact(note)
        decision: dict[str, Any] = {"action": action, "note": clean_note}
        warnings: list[str] = []
        if action == "edit":
            decision["reply"], _ = redact(reply.strip())
            warnings = quality_violations(decision["reply"])
        started = time.perf_counter()
        try:
            with tracing_context(enabled=self.settings.tracing, client=self._ls_client,
                                 project_name=self.settings.langsmith_project):
                state = self._traced_resume(ticket_id, decision, langsmith_extra={
                    "metadata": self._trace_meta(ticket_id, {"review_action": action,
                                                             "reviewer": reviewer})})
        except Exception:
            self.store.release_review(ticket_id)
            raise
        final = self._result(ticket_id, state, "ok", result["run_id"], started,
                             created_at=result["created_at"])
        final.update(history=result.get("history", []), pii_removed=result["pii_removed"],
                     latency_ms=result["latency_ms"], review_warnings=warnings,
                     review={"action": action, "note": clean_note, "reviewer": reviewer,
                             "reason": result.get("review_reason", ""),
                             "draft": result["reply"], "at": now()})
        self.store.update(ticket_id, final)
        self.store.add_review(ticket_id, action, result.get("review_reason", ""), clean_note,
                              result["reply"], final["reply"], reviewer)
        if self.checkpointer is not None:
            try:
                self.checkpointer.delete_thread(ticket_id)  # done; keep the file small
            except Exception:  # pragma: no cover - housekeeping only
                pass
        log.info("ticket reviewed", extra={"ticket_id": ticket_id, "action": action})
        return {**final, "replayed": False}

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
                "retrieval": self.retriever.name, "review_mode": self.settings.review_mode,
                "tracing": self.settings.tracing, "free_slots": self._slots._value}
