"""Versioned prompts live as plain markdown files next to this module.

A prompt change is a file change, which means it shows up in a pull request diff and
can be gated by `cx-gate` before it merges.
"""

from __future__ import annotations

from importlib import resources

DEFAULT_REPLY_PROMPT = "reply_v1_baseline"
# Chosen from real runs (qwen2.5:7b, 31 tickets x 3 repeats): v2 + rules had no missed and no
# unnecessary escalations; v1 + model missed an angry customer in 2 of 3 runs.
TRIAGE_PROMPT = "triage_v2"
DEFAULT_ESCALATION = "rules"


def available() -> list[str]:
    root = resources.files(__package__)
    return sorted(p.name[:-3] for p in root.iterdir() if p.name.endswith(".md"))


def load(name: str) -> str:
    path = resources.files(__package__) / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"No prompt named {name!r}. Available: {', '.join(available())}")
    return path.read_text(encoding="utf-8")
