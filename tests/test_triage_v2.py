import json

import pytest

from cx_copilot import compare as cmp
from cx_copilot import prompts
from cx_copilot.cli import main as cli_main
from cx_copilot.config import Settings
from cx_copilot.dataset import load_tickets
from cx_copilot.graph import _flag, build_graph, decide_escalation, parse_triage, run_ticket
from cx_copilot.models import CATEGORIES

CALM = json.dumps({"category": "billing", "urgency": "normal", "sentiment": "neutral",
                   "escalate": True})
HOT = json.dumps({"category": "shipping", "urgency": "high", "sentiment": "angry",
                  "escalate": False})
CLEAN = ("I'm sorry for the trouble. Duplicate charges are reversed in 3-5 business days. "
         "Next step: reply with your order number. - Northwind Support")


# ------------------------------------------------------------ parsing
@pytest.mark.parametrize("value,expected", [(True, True), (False, False), ("false", False),
                                            ("False", False), ("true", True), ("yes", True),
                                            ("no", False), (None, None), ("maybe", None)])
def test_flags_are_read_correctly(value, expected):
    assert _flag(value) is expected


def test_string_false_no_longer_counts_as_escalate():
    """Regression: bool("false") is True, which escalated every ticket for some models."""
    t = parse_triage('{"category": "billing", "urgency": "normal", "sentiment": "neutral", '
                     '"escalate": "false"}')
    assert t["escalate"] is False


def test_labels_are_normalised():
    t = parse_triage('{"category": "Safety/Legal", "urgency": "HIGH", "sentiment": "Mad"}')
    assert t == {"category": "safety_legal", "urgency": "high", "sentiment": "neutral",
                 "escalate": None}
    assert parse_triage('{"category": "Billing ", "urgency": "medium"}')["urgency"] == "normal"
    assert parse_triage('{"category": "weather"}') is None


def test_escalation_policy():
    assert decide_escalation({"category": "safety_legal", "urgency": "low", "sentiment": "neutral"})
    assert decide_escalation({"category": "billing", "urgency": "high", "sentiment": "angry"})
    assert not decide_escalation({"category": "billing", "urgency": "high", "sentiment": "frustrated"})
    assert not decide_escalation({"category": "billing", "urgency": "normal", "sentiment": "angry"})


# ------------------------------------------------------------ graph modes
def test_rules_mode_ignores_a_model_that_escalates_a_calm_ticket(scripted):
    m = scripted(CALM, [CLEAN])
    out = run_ticket(build_graph(m, escalation="rules"), "I was charged twice")
    assert not out["escalated"] and out["reply"] == CLEAN


def test_rules_mode_escalates_angry_urgent_even_if_the_model_says_no(scripted):
    out = run_ticket(build_graph(scripted(HOT, [CLEAN]), escalation="rules"), "late order")
    assert out["escalated"]


def test_model_mode_still_trusts_the_flag(scripted):
    out = run_ticket(build_graph(scripted(CALM, [CLEAN]), escalation="model"), "charged twice")
    assert out["escalated"]


def test_defaults_are_the_configuration_chosen_from_real_runs():
    assert prompts.TRIAGE_PROMPT == "triage_v2" and prompts.DEFAULT_ESCALATION == "rules"
    assert Settings().triage_prompt == "triage_v2" and Settings().escalation == "rules"


def test_a_prompt_without_an_escalate_field_falls_back_to_rules(scripted):
    no_flag = json.dumps({"category": "billing", "urgency": "high", "sentiment": "angry"})
    out = run_ticket(build_graph(scripted(no_flag, [CLEAN]), triage_prompt="triage_v2"), "x")
    assert out["escalated"]


def test_bad_escalation_mode_is_rejected(scripted):
    with pytest.raises(ValueError, match="escalation"):
        build_graph(scripted(CALM, [CLEAN]), escalation="vibes")


def test_safety_regex_still_beats_rules_mode(scripted):
    out = run_ticket(build_graph(scripted(CALM, [CLEAN]), escalation="rules"),
                     "my charger is smoking")
    assert out["escalated"] and out["triage"]["category"] == "safety_legal"


# ------------------------------------------------------------ the prompt itself
def test_triage_v2_defines_every_label_and_does_not_ask_for_escalate():
    text = prompts.load("triage_v2")
    for label in (*CATEGORIES, "low", "normal", "high", "angry", "frustrated", "neutral", "positive"):
        assert f"{label}" in text
    assert '"escalate"' not in text


def test_triage_v2_examples_do_not_leak_eval_tickets():
    """Contamination guard: if an eval ticket appears in the prompt, the eval is meaningless."""
    text = prompts.load("triage_v2").lower()
    for row in load_tickets():
        ticket = row["ticket"].lower()
        for i in range(0, max(1, len(ticket) - 40), 10):
            assert ticket[i:i + 40] not in text, f"{row['id']} leaks into triage_v2"


# ------------------------------------------------------------ compare + config + cli
def test_variant_specs_parse_and_validate():
    assert cmp.parse_spec("ollama:qwen2.5:7b@triage_v2/rules", "triage_v1", "model") == \
        ("ollama:qwen2.5:7b", "triage_v2", "rules", "reply_v4_grounded")
    assert cmp.parse_spec("ollama:llama3.2", "triage_v1", "model") == \
        ("ollama:llama3.2", "triage_v1", "model", "reply_v4_grounded")
    assert cmp.parse_spec("ollama:qwen2.5:7b+reply_v4_grounded", "triage_v2", "rules")[3] == \
        "reply_v4_grounded"
    with pytest.raises(FileNotFoundError):
        cmp.parse_spec("fake+reply_nope", "triage_v1", "model")
    assert cmp.parse_spec("fake@triage_v2", "triage_v1", "model")[2] == "model"
    with pytest.raises(ValueError):
        cmp.parse_spec("fake@triage_v2/vibes", "triage_v1", "model")
    with pytest.raises(FileNotFoundError):
        cmp.parse_spec("fake@nope", "triage_v1", "model")


def test_qa_fallback_escalations_are_labelled_as_such(scripted, monkeypatch):
    from cx_copilot import runner
    from cx_copilot.models import FakeModel

    class BadDrafts(FakeModel):  # good triage, replies that always fail QA
        def complete(self, task, system, user, *, context=None):
            if task == "reply":
                return "We guarantee it."
            return super().complete(task, system, user, context=context)

    monkeypatch.setattr(runner, "get_model", lambda spec, **kw: BadDrafts())
    report = cmp.run_compare(["fake", "fake@triage_v2/rules"], "reply_v1_baseline")
    misses = report["results"]["fake"]["misses"]
    t01 = misses["t01"]
    assert t01["counts"]["over_escalation"] == 1 and t01["example"]["escalated_by"] == "qa"
    assert "QA fallback" in cmp.render(report)


def test_compare_lists_the_exact_missed_escalations(tmp_path):
    report = cmp.run_compare(["fake", "fake:lite"], "reply_v1_baseline")
    misses = report["results"]["fake:lite"]["misses"]
    missed = sorted(t for t, m in misses.items() if m["counts"]["missed_escalation"])
    # fake:lite ignores tone and urgency, so it misses every angry + urgent ticket
    # (third contacts in a thread still escalate: that rule does not depend on the model)
    assert missed == ["m19", "t20", "t21", "t31", "t97", "t98"]
    md = cmp.render(report)
    assert "Missed escalations" in md and "`t20` (1/1)" in md


def test_compare_cli_accepts_variants_and_rejects_bad_ones(tmp_path):
    out = tmp_path / "o"
    assert cmp.main(["--models", "fake", "fake@triage_v2/rules", "--out", str(out)]) == 0
    data = json.loads(out.with_suffix(".json").read_text())
    assert set(data["results"]) == {"fake", "fake@triage_v2/rules"}
    assert cmp.main(["--models", "fake", "fake@nope", "--out", str(out)]) == 2


def test_service_settings_validate_the_new_switches(monkeypatch):
    monkeypatch.setenv("CX_ESCALATION", "vibes")
    with pytest.raises(ValueError, match="CX_ESCALATION"):
        Settings.from_env()
    monkeypatch.setenv("CX_ESCALATION", "rules")
    monkeypatch.setenv("CX_TRIAGE_PROMPT", "nope")
    with pytest.raises(FileNotFoundError):
        Settings.from_env()


def test_cli_runs_with_v2_and_rules(capsys):
    assert cli_main(["I was charged twice", "--triage-prompt", "triage_v2",
                     "--escalation", "rules"]) == 0
    assert "auto-reply" in capsys.readouterr().out
