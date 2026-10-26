import json

import pytest

from cx_copilot import prompts
from cx_copilot.models import FakeModel, get_model, keyword_triage


def test_fake_triage_is_deterministic():
    m = FakeModel()
    a = m.complete("triage", "", "I was charged twice", context={"ticket": "I was charged twice"})
    b = m.complete("triage", "", "I was charged twice", context={"ticket": "I was charged twice"})
    assert a == b and json.loads(a)["category"] == "billing"


def test_safety_words_force_safety_legal():
    assert keyword_triage("my charger is smoking")["category"] == "safety_legal"
    assert keyword_triage("I will call my lawyer")["escalate"] is True


def test_angry_and_urgent_escalates_but_urgent_alone_does_not():
    assert keyword_triage("This is UNACCEPTABLE, fix it immediately")["escalate"] is True
    assert keyword_triage("Urgent, please help me log in ASAP")["escalate"] is False


def test_lite_model_ignores_tone_and_urgency():
    t = keyword_triage("This is UNACCEPTABLE, fix it immediately", lite=True)
    assert t["escalate"] is False and t["urgency"] == "normal"


def test_reply_changes_with_prompt_directives():
    ctx = {"triage": {"category": "returns", "sentiment": "neutral"},
           "policies": ["Refunds take 5-7 business days."], "feedback": ""}
    full = FakeModel().complete("reply", prompts.load("reply_v1_baseline"), "", context=ctx)
    loose = FakeModel().complete("reply", prompts.load("reply_v3_friendly"), "", context=ctx)
    assert "5-7 business days" in full
    assert "5-7 business days" not in loose


def test_get_model_specs():
    assert get_model("fake").name == "fake"
    assert get_model("fake:lite").name == "fake:lite"
    for bad in ("nope", "anthropic:", "openai"):
        with pytest.raises(ValueError):
            get_model(bad)
