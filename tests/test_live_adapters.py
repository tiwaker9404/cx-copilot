"""The live adapters, exercised with stub SDK clients (no network, no keys).

These prove the request shape, parsing, caching, error handling and end-to-end wiring.
They do NOT prove anything about real model behaviour; only a live run can.
"""

import json
from types import SimpleNamespace

import pytest

from cx_copilot import compare as cmp
from cx_copilot import runner
from cx_copilot.dataset import load_tickets
from cx_copilot.gate import main as gate_main
from cx_copilot.models import (AnthropicModel, CachedModel, FakeModel, OpenAIModel, get_model,
                               keyword_triage, missing_key)
from cx_copilot.runner import RunConfig, run_suite


class StubAnthropic:
    """Answers like a decent model: triage JSON for the triage prompt, a clean reply otherwise."""

    def __init__(self, fail_after=None):
        self.calls, self.fail_after = [], fail_after
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise ConnectionError("simulated outage")
        user = kw["messages"][0]["content"]
        if kw["system"].startswith("You triage"):
            text = json.dumps(keyword_triage(user))
        else:
            notes = user.split("Policy notes:\n", 1)[-1].replace("- ", "")
            text = ("I'm sorry about the trouble. " + notes.strip() +
                    " Next step: reply with your order number. - Northwind Support")
        return SimpleNamespace(content=[SimpleNamespace(type="thinking", text="ignore me"),
                                        SimpleNamespace(type="text", text=text)])


def test_anthropic_request_shape_and_text_only_parsing():
    stub = StubAnthropic()
    out = AnthropicModel("m-1", client=stub).complete("reply", "SYS", "hello")
    kw = stub.calls[0]
    assert kw["model"] == "m-1" and kw["system"] == "SYS" and kw["max_tokens"] > 0
    assert kw["messages"] == [{"role": "user", "content": "hello"}]
    assert "ignore me" not in out


def test_openai_request_shape_and_empty_content():
    seen = {}

    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    stub = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert OpenAIModel("m-2", client=stub).complete("reply", "SYS", "hi") == ""
    assert seen["messages"][0] == {"role": "system", "content": "SYS"}


def test_full_suite_through_the_live_adapter_path():
    stub = StubAnthropic()
    cfg = RunConfig("anthropic:stub", client=stub)
    report = run_suite(cfg)
    assert report["errors"] == 0 and report["n"] == len(load_tickets())
    assert report["metrics"]["safety"] == 1.0
    assert len(stub.calls) >= report["n"]          # at least one model call per ticket
    assert report["config"] == {"model": "anthropic:stub", "prompt": cfg.prompt}


def test_cache_makes_reruns_free_and_repeats_make_fresh_calls(tmp_path):
    stub = StubAnthropic()
    first = run_suite(RunConfig("anthropic:stub", cache_dir=str(tmp_path), client=stub))
    n = len(stub.calls)
    again = run_suite(RunConfig("anthropic:stub", cache_dir=str(tmp_path), client=stub))
    assert len(stub.calls) == n and again["metrics"] == first["metrics"]
    run_suite(RunConfig("anthropic:stub", cache_dir=str(tmp_path), repeat=1, client=stub))
    assert len(stub.calls) == 2 * n               # a new repeat index is a new set of calls


def test_cache_wraps_only_live_models_and_survives_corruption_free_writes(tmp_path):
    assert isinstance(get_model("fake", cache_dir=tmp_path), FakeModel)
    m = CachedModel(AnthropicModel("m", client=StubAnthropic()), tmp_path)
    m.complete("reply", "s", "u")
    m.complete("reply", "s", "u")
    assert (m.hits, m.misses) == (1, 1)
    assert not list(tmp_path.glob("*.tmp"))


def test_one_failing_call_does_not_sink_the_run_but_is_counted():
    stub = StubAnthropic(fail_after=20)
    report = run_suite(RunConfig("anthropic:stub", client=stub))
    assert 0 < report["errors"] < report["n"]
    bad = [t for t in report["tickets"].values() if t["error"]]
    assert len(bad) == report["errors"] and "ConnectionError" in bad[0]["error"]
    assert report["metrics"]["escalation_recall"] < 1.0   # failures are penalised, not hidden


def test_gate_refuses_to_judge_a_run_with_errors(monkeypatch, tmp_path):
    base = tmp_path / "b.json"
    assert gate_main(["--baseline", str(base), "--update-baseline"]) == 0
    stub = StubAnthropic(fail_after=5)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(runner, "get_model",
                        lambda spec, **kw: AnthropicModel("m", client=stub) if spec != "fake"
                        else FakeModel())
    assert gate_main(["--baseline", str(base), "--model", "anthropic:m"]) == 2


def test_missing_api_keys_are_caught_before_any_work(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert missing_key("anthropic:x") == "ANTHROPIC_API_KEY"
    assert missing_key("openai:x") == "OPENAI_API_KEY" and missing_key("fake") is None
    assert gate_main(["--baseline", str(tmp_path / "b.json"), "--model", "anthropic:x"]) == 2
    assert cmp.main(["--models", "fake", "anthropic:x", "--out", str(tmp_path / "o")]) == 2


def test_compare_flags_the_swap_the_gate_would_block(tmp_path):
    report = cmp.run_compare(["fake", "fake:lite"], "reply_v1_baseline")
    verdict = report["verdicts"]["fake:lite"]
    assert not verdict["passed"] and "escalation_recall" in verdict["failed_metrics"]
    md = cmp.render(report)
    assert "BLOCKED on" in md and "offline stand-ins" in md


def test_compare_cli_writes_markdown_and_json(tmp_path):
    out = tmp_path / "result"
    assert cmp.main(["--models", "fake", "fake:lite", "--out", str(out)]) == 0
    assert out.with_suffix(".md").exists()
    assert json.loads(out.with_suffix(".json").read_text())["reference"] == "fake"


def test_compare_cli_rejects_one_model_and_bad_specs(tmp_path):
    assert cmp.main(["--models", "fake", "--out", str(tmp_path / "o")]) == 2
    assert cmp.main(["--models", "fake", "wat", "--out", str(tmp_path / "o")]) == 2


def test_repeats_are_averaged_and_their_spread_reported():
    def run(q):
        return {"metrics": {"quality": q}, "errors": 0, "seconds": 1.0, "n": 1,
                "tickets": {"t1": {"scores": {"quality": q}}}}

    agg = cmp._aggregate([run(0.8), run(1.0)])
    assert agg["metrics"]["quality"] == 0.9 and agg["spread"]["quality"] == 0.2
    assert agg["tickets"]["t1"]["scores"]["quality"] == 0.9 and agg["repeats"] == 2
