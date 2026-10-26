"""Service configuration, read once from environment variables and validated at startup.

A bad setting should stop the service from booting, not fail on the first customer ticket.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import kb, prompts


def _int(name: str, default: int, low: int = 1) -> int:
    raw = os.environ.get(name, "")
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < low:
        raise ValueError(f"{name} must be >= {low}, got {value}")
    return value


def _tracing_on() -> bool:
    flag = (os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2") or "")
    return flag.lower() in ("1", "true", "yes") and bool(
        os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))


@dataclass(frozen=True)
class Settings:
    env: str = "dev"                                  # dev | prod
    model: str = "fake"
    reply_prompt: str = prompts.DEFAULT_REPLY_PROMPT
    triage_prompt: str = prompts.TRIAGE_PROMPT
    escalation: str = prompts.DEFAULT_ESCALATION
    api_keys: tuple[str, ...] = ()
    db_path: str = "data/cx.db"
    kb_path: str | None = None
    max_ticket_chars: int = 4000
    max_concurrency: int = 4                          # parallel model calls (protects a local model)
    queue_timeout_s: int = 30                         # wait this long for a free slot, then 503
    rate_limit_per_min: int = 60                      # per API key
    retrieval: str = kb.DEFAULT_METHOD
    embed_model: str | None = None                    # e.g. ollama:nomic-embed-text
    review_mode: str = "off"                          # off | risky | all
    checkpoint_path: str = "data/checkpoints.db"      # paused reviews survive restarts
    max_history: int = 20                             # earlier messages accepted per ticket
    tracing: bool = False
    langsmith_project: str = "cx-copilot"
    docs_enabled: bool = True
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get("CX_ENV", "dev").lower()
        if env not in ("dev", "prod"):
            raise ValueError(f"CX_ENV must be dev or prod, got {env!r}")
        keys = tuple(k.strip() for k in os.environ.get("CX_API_KEYS", "").split(",") if k.strip())
        s = cls(
            env=env,
            model=os.environ.get("CX_MODEL", "fake"),
            reply_prompt=os.environ.get("CX_REPLY_PROMPT", prompts.DEFAULT_REPLY_PROMPT),
            triage_prompt=os.environ.get("CX_TRIAGE_PROMPT", prompts.TRIAGE_PROMPT),
            escalation=os.environ.get("CX_ESCALATION", prompts.DEFAULT_ESCALATION),
            api_keys=keys,
            db_path=os.environ.get("CX_DB_PATH", "data/cx.db"),
            kb_path=os.environ.get("CX_KB_PATH") or None,
            max_ticket_chars=_int("CX_MAX_TICKET_CHARS", 4000, 100),
            max_concurrency=_int("CX_MAX_CONCURRENCY", 4),
            queue_timeout_s=_int("CX_QUEUE_TIMEOUT_S", 30),
            rate_limit_per_min=_int("CX_RATE_LIMIT_PER_MIN", 60),
            retrieval=os.environ.get("CX_RETRIEVAL", kb.DEFAULT_METHOD),
            embed_model=os.environ.get("CX_EMBED_MODEL") or None,
            review_mode=os.environ.get("CX_REVIEW_MODE", "off").lower(),
            checkpoint_path=os.environ.get("CX_CHECKPOINT_PATH", "data/checkpoints.db"),
            max_history=_int("CX_MAX_HISTORY", 20, 0),
            tracing=_tracing_on(),
            langsmith_project=os.environ.get("LANGSMITH_PROJECT", "cx-copilot"),
            docs_enabled=os.environ.get("CX_DOCS", "1" if env == "dev" else "0") == "1",
        )
        s.validate()
        return s

    def validate(self) -> None:
        prompts.load(self.reply_prompt)  # raises on an unknown prompt name
        prompts.load(self.triage_prompt)
        if self.escalation not in ("model", "rules"):
            raise ValueError(f"CX_ESCALATION must be model or rules, got {self.escalation!r}")
        if self.retrieval not in kb.METHODS:
            raise ValueError(f"CX_RETRIEVAL must be one of {', '.join(kb.METHODS)}, "
                             f"got {self.retrieval!r}")
        if self.retrieval == "embed" and not self.embed_model:
            raise ValueError("CX_RETRIEVAL=embed needs CX_EMBED_MODEL, e.g. ollama:nomic-embed-text")
        if self.review_mode not in ("off", "risky", "all"):
            raise ValueError(f"CX_REVIEW_MODE must be off, risky or all, got {self.review_mode!r}")
        if self.env == "prod":
            if not self.api_keys:
                raise ValueError("CX_ENV=prod requires CX_API_KEYS (comma-separated)")
            if any(len(k) < 16 for k in self.api_keys):
                raise ValueError("every key in CX_API_KEYS must be at least 16 characters")
            if self.model.startswith("fake"):
                raise ValueError("CX_ENV=prod with a fake model: set CX_MODEL, e.g. ollama:llama3.2")

    @property
    def auth_required(self) -> bool:
        return self.env == "prod" or bool(self.api_keys)
