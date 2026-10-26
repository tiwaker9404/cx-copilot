import json
from types import SimpleNamespace

from cx_copilot import gate
from cx_copilot.models import CachedModel, FakeModel, OpenAIModel, pop_usage
from cx_copilot.runner import RunConfig, percentile, run_suite


def test_openai_adapter_records_reported_usage():
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
                           usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30))
    stub = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: resp)))
    OpenAIModel("x", stub).complete("reply", "s", "u")
    u = pop_usage()
    assert (u["input_tokens"], u["output_tokens"], u["estimated"]) == (120, 30, False)
    assert u["latency_ms"] >= 0


def test_cache_replays_recorded_usage(tmp_path):
    m = CachedModel(FakeModel(), tmp_path)
    m.complete("reply", "never promise", "ticket", context={"ticket": "hi"})
    first = pop_usage()
    m.complete("reply", "never promise", "ticket", context={"ticket": "hi"})
    replay = pop_usage()
    assert replay["cached"] and replay["input_tokens"] == first["input_tokens"]


def test_old_cache_entries_without_usage_still_work(tmp_path):
    m = CachedModel(FakeModel(), tmp_path)
    path = m._path("reply", "s", "u")
    tmp_path.mkdir(exist_ok=True)
    path.write_text(json.dumps({"model": "fake", "task": "reply", "text": "old"}))
    assert m.complete("reply", "s", "u") == "old"
    u = pop_usage()
    assert u["cached"] and u["estimated"] and u["latency_ms"] is None


def test_suite_reports_perf():
    report = run_suite(RunConfig())
    perf = report["perf"]
    assert perf["tickets"] == report["n"]
    assert 1 <= perf["mean_calls_per_ticket"] <= 3
    assert perf["mean_tokens_per_ticket"] > 100


def test_percentile():
    assert percentile([], 95) is None
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile(list(range(1, 101)), 95) == 95


def test_budget_check_and_model_specific_limits():
    budgets = {"default": {"mean_calls_per_ticket": 2.5, "mean_tokens_per_ticket": 3000},
               "ollama:big": {"mean_tokens_per_ticket": 1000}}
    perf = {"mean_calls_per_ticket": 2.0, "mean_tokens_per_ticket": 1500,
            "p95_model_ms_per_ticket": 100}
    rows = {r["budget"]: r["status"] for r in gate.check_budgets(perf, gate.budget_for(budgets, "fake"))}
    assert rows == {"mean_calls_per_ticket": "ok", "mean_tokens_per_ticket": "ok",
                    "p95_model_ms_per_ticket": "SKIP"}
    big = gate.check_budgets(perf, gate.budget_for(budgets, "ollama:big"))
    assert [r["status"] for r in big][:2] == ["ok", "FAIL"]


def test_gate_blocks_on_a_blown_budget(tmp_path):
    tight = tmp_path / "budgets.json"
    tight.write_text(json.dumps({"default": {"mean_tokens_per_ticket": 10}}))
    assert gate.main(["--budgets", str(tight)]) == 1
    assert gate.main(["--budgets", str(tight), "--no-budgets"]) == 0
