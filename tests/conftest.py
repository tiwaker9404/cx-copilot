import os

import pytest

# Never send traces from the test suite, whatever the developer's shell has exported.
for var in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
    os.environ.pop(var, None)
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


class ScriptedModel:
    """Test double: returns canned triage JSON and a queue of reply drafts."""

    name = "scripted"

    def __init__(self, triage: str, replies: list[str] | None = None) -> None:
        self.triage = triage
        self.replies = list(replies or [])
        self.reply_calls = 0

    def complete(self, task, system, user, *, context=None):
        if task == "triage":
            return self.triage
        self.reply_calls += 1
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


@pytest.fixture
def scripted():
    return ScriptedModel
