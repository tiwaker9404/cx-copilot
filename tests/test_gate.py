import json
from pathlib import Path

import pytest

from cx_copilot.gate import MAX_DROP, compare, main

REPO_BASELINE = Path(__file__).resolve().parents[1] / "evals" / "baseline.json"


def payload(**metrics):
    base = {m: 1.0 for m in MAX_DROP}
    base.update(metrics)
    return {"metrics": base, "ticket_quality": {"t01": 1.0}}


def candidate(**metrics):
    p = payload(**metrics)
    return {"metrics": p["metrics"], "tickets": {"t01": {"scores": {"quality": 0.5}}}}


def test_identical_scores_pass():
    assert compare(payload(), candidate())["passed"]


def test_small_drop_inside_tolerance_passes():
    assert compare(payload(), candidate(empathy=0.96))["passed"]


def test_drop_beyond_tolerance_fails():
    res = compare(payload(), candidate(policy_grounded=0.80))
    assert not res["passed"]
    assert [r["metric"] for r in res["rows"] if r["status"] == "FAIL"] == ["policy_grounded"]


def test_zero_tolerance_metrics_block_on_any_drop():
    assert not compare(payload(), candidate(safety=0.99))["passed"]
    assert not compare(payload(), candidate(escalation_recall=0.99))["passed"]


def test_average_can_hide_a_collapse_that_the_gate_still_catches():
    res = compare(payload(), candidate(quality=0.99, escalation_recall=0.5))
    assert not res["passed"]


def test_improvements_pass():
    assert compare(payload(empathy=0.5), candidate(empathy=1.0))["passed"]


def test_missing_metric_is_skipped_not_failed():
    cand = candidate()
    del cand["metrics"]["empathy"]
    res = compare(payload(), cand)
    assert res["passed"] and any(r["status"] == "SKIP" for r in res["rows"])


def test_per_ticket_regressions_are_listed():
    res = compare(payload(), candidate())
    assert res["regressed_tickets"][0]["id"] == "t01"


# ---- end-to-end through the CLI entry point (offline, fake model) ----

def test_cli_missing_baseline_is_a_setup_error(tmp_path, capsys):
    assert main(["--baseline", str(tmp_path / "nope.json")]) == 2


def test_cli_unknown_prompt_is_a_setup_error(tmp_path):
    assert main(["--baseline", str(tmp_path / "b.json"), "--prompt", "does_not_exist"]) == 2


def test_cli_full_cycle(tmp_path, capsys):
    b = tmp_path / "b.json"
    assert main(["--baseline", str(b), "--update-baseline"]) == 0  # create
    assert main(["--baseline", str(b)]) == 0                       # same config passes
    assert main(["--baseline", str(b), "--prompt", "reply_v2_tighter"]) == 0
    before = b.read_text()
    assert main(["--baseline", str(b), "--prompt", "reply_v3_friendly"]) == 1   # blocked
    assert main(["--baseline", str(b), "--model", "fake:lite"]) == 1            # blocked
    # a failing candidate must never overwrite the baseline
    assert main(["--baseline", str(b), "--prompt", "reply_v3_friendly",
                 "--update-baseline"]) == 1
    assert b.read_text() == before
    assert "BLOCKED" in capsys.readouterr().out


def test_cli_json_output_is_parseable(tmp_path, capsys):
    b = tmp_path / "b.json"
    main(["--baseline", str(b), "--update-baseline"])
    capsys.readouterr()
    assert main(["--baseline", str(b), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["passed"] is True


@pytest.mark.skipif(not REPO_BASELINE.exists(), reason="no committed baseline")
def test_committed_baseline_matches_current_code(monkeypatch):
    """If this fails, the code changed behaviour; review it, then `cx-gate --update-baseline`."""
    monkeypatch.chdir(REPO_BASELINE.parents[1])
    assert main([]) == 0
