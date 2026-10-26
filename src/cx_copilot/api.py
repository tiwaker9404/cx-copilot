"""HTTP API for cx-copilot.

    cx-serve                         # uvicorn on 0.0.0.0:8000, settings from the environment

    POST /v1/tickets                 handle a ticket (idempotent on ticket_id)
    GET  /v1/tickets/{ticket_id}     fetch a stored result
    POST /v1/tickets/{id}/feedback   rate a reply; forwarded to the LangSmith trace
    GET  /v1/escalations             the human queue
    GET  /v1/stats                   counts for a dashboard
    GET  /healthz  /readyz           liveness and readiness probes
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .service import Busy, Conflict, NotFound, TicketService

log = logging.getLogger("cx_copilot")

_STD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None)))


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Ticket text is never passed to the logger."""

    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
               "msg": record.getMessage()}
        out.update({k: v for k, v in vars(record).items() if k not in _STD})
        return json.dumps(out, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    log.handlers[:] = [handler]
    log.setLevel(level)
    log.propagate = False


# ------------------------------------------------------------------ schemas
class TicketIn(BaseModel):
    text: str = Field(min_length=1, description="The customer's message")
    ticket_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.:-]{1,64}$",
                                  description="Your id; resending the same id is safe")


class FeedbackIn(BaseModel):
    score: float = Field(ge=0, le=1, description="0 = bad reply, 1 = good reply")
    comment: str = Field(default="", max_length=1000)


class TicketOut(BaseModel):
    ticket_id: str
    status: str
    category: str
    urgency: str
    sentiment: str
    escalated: bool
    reply: str
    handoff: str
    policy_ids: list[str]
    attempts: int
    pii_removed: list[str]
    model: str
    reply_prompt: str
    run_id: str
    traced: bool
    latency_ms: int
    created_at: str
    replayed: bool = False


# --------------------------------------------------------------- rate limit
class RateLimiter:
    """Sliding one-minute window per caller. In-process: run one worker, or put a shared
    limiter (for example at the reverse proxy) in front if you scale out."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, who: str) -> bool:
        cutoff = time.monotonic() - 60
        with self._lock:
            q = self._hits[who]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(time.monotonic())
            return True


# --------------------------------------------------------------------- app
def create_app(settings: Settings | None = None, service: TicketService | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or TicketService(settings)
    limiter = RateLimiter(settings.rate_limit_per_min)
    app = FastAPI(title="cx-copilot", version=__version__,
                  docs_url="/docs" if settings.docs_enabled else None, redoc_url=None,
                  openapi_url="/openapi.json" if settings.docs_enabled else None)
    app.state.service = service
    app.state.settings = settings

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        log.info("request", extra={"request_id": rid, "method": request.method,
                                   "path": request.url.path, "status": response.status_code,
                                   "ms": int((time.perf_counter() - started) * 1000)})
        return response

    def caller(x_api_key: str | None = Header(default=None)) -> str:
        if settings.auth_required:
            if not x_api_key or not any(hmac.compare_digest(x_api_key, k)
                                        for k in settings.api_keys):
                raise HTTPException(401, "missing or invalid X-API-Key")
            who = "key:" + x_api_key[-4:]
        else:
            who = "anonymous"
        if not limiter.allow(who):
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "60"})
        return who

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        info = service.ready()
        return JSONResponse(info, status_code=200 if info["store"] else 503)

    @app.post("/v1/tickets", response_model=TicketOut)
    def create_ticket(body: TicketIn, _: str = Depends(caller)) -> dict[str, Any]:
        if len(body.text) > settings.max_ticket_chars:
            raise HTTPException(413, f"ticket longer than {settings.max_ticket_chars} characters")
        try:
            return service.handle(body.text, body.ticket_id)
        except Conflict:
            raise HTTPException(409, "ticket_id already used with different text") from None
        except Busy:
            raise HTTPException(503, "all model slots busy, retry shortly",
                                headers={"Retry-After": "5"}) from None

    @app.get("/v1/tickets/{ticket_id}", response_model=TicketOut)
    def get_ticket(ticket_id: str, _: str = Depends(caller)) -> dict[str, Any]:
        try:
            return service.get(ticket_id)
        except NotFound:
            raise HTTPException(404, "unknown ticket_id") from None

    @app.post("/v1/tickets/{ticket_id}/feedback")
    def feedback(ticket_id: str, body: FeedbackIn, _: str = Depends(caller)) -> dict[str, Any]:
        try:
            return service.feedback(ticket_id, body.score, body.comment)
        except NotFound:
            raise HTTPException(404, "unknown ticket_id") from None

    @app.get("/v1/escalations", response_model=list[TicketOut])
    def escalations(limit: int = Query(50, ge=1, le=500), _: str = Depends(caller)):
        return service.escalations(limit)

    @app.get("/v1/stats")
    def stats(_: str = Depends(caller)) -> dict[str, Any]:
        return service.store.stats()

    return app


def main() -> None:  # pragma: no cover - thin wrapper around uvicorn
    import uvicorn

    configure_logging(os.environ.get("CX_LOG_LEVEL", "INFO"))
    try:
        Settings.from_env()  # fail fast with a readable message before uvicorn starts
    except (ValueError, FileNotFoundError) as exc:
        print(f"cx-serve: configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    uvicorn.run("cx_copilot.api:create_app", factory=True,
                host=os.environ.get("CX_HOST", "0.0.0.0"),
                port=int(os.environ.get("CX_PORT", "8000")),
                workers=1, access_log=False, proxy_headers=True)
