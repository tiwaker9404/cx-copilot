"""Durable storage for handled tickets and feedback. SQLite: zero cost, zero infrastructure.

One short-lived connection per operation keeps it safe across FastAPI's worker threads. WAL mode
lets reads continue while a write is in progress. Only redacted ticket text is ever stored.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id    TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    text_hash    TEXT NOT NULL,
    ticket       TEXT NOT NULL,
    result       TEXT NOT NULL,
    escalated    INTEGER NOT NULL,
    status       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tickets_escalated ON tickets (escalated, created_at);
CREATE INDEX IF NOT EXISTS tickets_status ON tickets (status, created_at);
CREATE TABLE IF NOT EXISTS reviews (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id    TEXT NOT NULL REFERENCES tickets (ticket_id),
    created_at   TEXT NOT NULL,
    action       TEXT NOT NULL,
    reason       TEXT NOT NULL,
    note         TEXT NOT NULL,
    draft        TEXT NOT NULL,
    final_reply  TEXT NOT NULL,
    reviewer     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id          TEXT NOT NULL REFERENCES tickets (ticket_id),
    created_at         TEXT NOT NULL,
    score              REAL NOT NULL,
    comment            TEXT NOT NULL,
    sent_to_langsmith  INTEGER NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()  # serialises writes; re-entrant because _conn also takes it
        self._memory = sqlite3.connect(path, check_same_thread=False) if path == ":memory:" else None
        with self._conn() as c:
            if self._memory is None:
                c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if self._memory is not None:
            with self._lock:
                yield self._memory
                self._memory.commit()
            return
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def ping(self) -> bool:
        try:
            with self._conn() as c:
                c.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def get(self, ticket_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT text_hash, result FROM tickets WHERE ticket_id = ?",
                            (ticket_id,)).fetchone()
        if row is None:
            return None
        return {"text_hash": row[0], "result": json.loads(row[1])}

    def save(self, ticket_id: str, text_hash: str, ticket: str, result: dict[str, Any]) -> bool:
        """Insert once. Returns False if the ticket id already exists (idempotent retries)."""
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ticket_id, result["created_at"], text_hash, ticket, json.dumps(result),
                 int(result["escalated"]), result["status"]))
            return cur.rowcount == 1

    def update(self, ticket_id: str, result: dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE tickets SET result = ?, escalated = ?, status = ? "
                      "WHERE ticket_id = ?", (json.dumps(result), int(result["escalated"]),
                                              result["status"], ticket_id))

    def claim_review(self, ticket_id: str) -> bool:
        """Atomically move a ticket from pending_review to reviewing, so two reviewers
        cannot resume the same paused run."""
        with self._lock, self._conn() as c:
            cur = c.execute("UPDATE tickets SET status = 'reviewing' WHERE ticket_id = ? AND "
                            "status = 'pending_review'", (ticket_id,))
            return cur.rowcount == 1

    def release_review(self, ticket_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE tickets SET status = 'pending_review' WHERE ticket_id = ? AND "
                      "status = 'reviewing'", (ticket_id,))

    def pending_reviews(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT result FROM tickets WHERE status = 'pending_review' "
                             "ORDER BY created_at ASC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def add_review(self, ticket_id: str, action: str, reason: str, note: str, draft: str,
                   final_reply: str, reviewer: str) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute("INSERT INTO reviews (ticket_id, created_at, action, reason, note, "
                            "draft, final_reply, reviewer) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (ticket_id, now(), action, reason, note, draft, final_reply,
                             reviewer))
            return int(cur.lastrowid)

    def harvest_rows(self, max_score: float = 0.5) -> list[dict[str, Any]]:
        """Reviewed-and-changed tickets plus low-rated ones, with their stored text."""
        with self._conn() as c:
            reviews = c.execute(
                "SELECT r.ticket_id, r.action, r.note, r.draft, r.final_reply, t.ticket, "
                "t.result FROM reviews r JOIN tickets t USING (ticket_id) "
                "WHERE r.action IN ('edit', 'reject') ORDER BY r.id").fetchall()
            low = c.execute(
                "SELECT f.ticket_id, f.score, f.comment, t.ticket, t.result FROM feedback f "
                "JOIN tickets t USING (ticket_id) WHERE f.score <= ? ORDER BY f.id",
                (max_score,)).fetchall()
        out = [{"ticket_id": r[0], "source": f"review_{r[1]}", "note": r[2], "draft": r[3],
                "final_reply": r[4], "ticket": r[5], "result": json.loads(r[6])}
               for r in reviews]
        out += [{"ticket_id": r[0], "source": "low_feedback", "score": r[1], "note": r[2],
                 "ticket": r[3], "result": json.loads(r[4])} for r in low]
        return out

    def escalations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT result FROM tickets WHERE escalated = 1 "
                             "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def add_feedback(self, ticket_id: str, score: float, comment: str, sent: bool) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute("INSERT INTO feedback (ticket_id, created_at, score, comment, "
                            "sent_to_langsmith) VALUES (?, ?, ?, ?, ?)",
                            (ticket_id, now(), score, comment, int(sent)))
            return int(cur.lastrowid)

    def stats(self) -> dict[str, Any]:
        with self._conn() as c:
            total, esc, degraded = c.execute(
                "SELECT COUNT(*), COALESCE(SUM(escalated), 0), "
                "COALESCE(SUM(status = 'degraded'), 0) FROM tickets").fetchone()
            pending = c.execute("SELECT COUNT(*) FROM tickets WHERE status IN "
                                "('pending_review', 'reviewing')").fetchone()[0]
            fb, avg = c.execute("SELECT COUNT(*), AVG(score) FROM feedback").fetchone()
            actions = dict(c.execute("SELECT action, COUNT(*) FROM reviews GROUP BY action")
                           .fetchall())
        reviewed = sum(actions.values())
        return {"tickets": total, "escalated": esc, "degraded": degraded,
                "pending_review": pending, "reviews": actions,
                "review_edit_rate": round(actions.get("edit", 0) / reviewed, 3) if reviewed
                else None,
                "feedback": fb, "avg_feedback": None if avg is None else round(avg, 3)}
