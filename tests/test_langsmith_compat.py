"""Scorers must be accepted unchanged as LangSmith evaluators.

`DynamicRunEvaluator` is the adapter `langsmith.evaluate` wraps plain functions in, so this
checks the `(outputs, reference_outputs)` signature against the installed LangSmith without
any network access.
"""

import datetime as dt
import uuid

from langsmith.evaluation.evaluator import DynamicRunEvaluator
from langsmith.schemas import Example, Run

from cx_copilot.dataset import load_tickets
from cx_copilot.runner import RunConfig, make_target
from cx_copilot.scorers import ALL_SCORERS


def test_every_scorer_works_as_a_langsmith_evaluator():
    now = dt.datetime.now(dt.timezone.utc)
    row = load_tickets()[0]
    outputs = make_target(RunConfig())({"ticket": row["ticket"]})
    run_id = uuid.uuid4()
    run = Run(id=run_id, name="cx_copilot_ticket", start_time=now, run_type="chain",
              trace_id=run_id, dotted_order=now.strftime("%Y%m%dT%H%M%S%fZ") + str(run_id),
              inputs={"ticket": row["ticket"]}, outputs=outputs)
    example = Example(id=uuid.uuid4(), dataset_id=uuid.uuid4(), created_at=now,
                      inputs={"ticket": row["ticket"]}, outputs=row["gold"])

    keys = {DynamicRunEvaluator(fn).evaluate_run(run, example).key for fn in ALL_SCORERS}
    assert keys == {fn(outputs, row["gold"])["key"] for fn in ALL_SCORERS}
    assert {"quality", "safety", "category_accuracy"} <= keys
