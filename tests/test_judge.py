import json

from cx_copilot import judge as j
from cx_copilot.models import FakeModel
from cx_copilot.runner import RunConfig, run_suite


def test_parse_verdict_handles_strings_fences_and_junk():
    v = j.parse_verdict('```json\n{"grounded": "true", "answers_question": false, '
                        '"tone_ok": true, "no_invented_commitments": "no", "reason": "x"}\n```')
    assert v == {"grounded": True, "answers_question": False, "tone_ok": True,
                 "no_invented_commitments": False, "reason": "x"}
    assert j.parse_verdict("I think it's fine") is None
    assert j.parse_verdict('{"something": 1}') is None


def test_cohen_kappa():
    assert j.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert j.cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert j.cohen_kappa([], []) is None


def test_labels_file_is_valid_and_balanced():
    rows = j.load_labels()
    assert len(rows) >= 40
    for key in ("grounded", "answers_question", "tone_ok", "no_invented_commitments"):
        fails = sum(not r["labels"][key] for r in rows)
        assert 4 <= fails <= len(rows) // 2, key


def test_calibration_report_and_cli(tmp_path):
    report = j.calibrate(j.Judge(FakeModel()), j.load_labels())
    assert report["n"] == 40 and report["unparseable"] == 0
    assert set(report["criteria"]) == {"grounded", "answers_question", "tone_ok",
                                       "no_invented_commitments"}
    out = tmp_path / "cal"
    assert j.main(["--out", str(out)]) == 0
    assert "Trusted" in out.with_suffix(".md").read_text()
    assert json.loads(out.with_suffix(".json").read_text())["judge"] == "fake/judge_v1"


def test_judge_metrics_appear_only_when_a_judge_is_used():
    rows = [{"id": "x", "ticket": "How do I cancel my subscription?",
             "gold": {"category": "billing", "escalate": False,
                      "required_facts": ["Settings > Billing"], "needs_empathy": False,
                      "relevant_policies": ["billing_subscription"]}}]
    plain = run_suite(RunConfig(prompt="reply_v4_grounded"), rows)
    judged = run_suite(RunConfig(prompt="reply_v4_grounded", judge="fake"), rows)
    assert not any(k.startswith("judge_") for k in plain["metrics"])
    assert judged["metrics"]["judge_no_invented_commitments"] == 1.0
    assert judged["config"]["judge"] == "fake"
