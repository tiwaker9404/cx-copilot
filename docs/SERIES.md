# Weekly series plan

One small, shippable upgrade per week. Each week = one PR, one tagged release, one LinkedIn post
with a number in the headline.

| Week | Ships | Headline number |
|---|---|---|
| 1 (this repo) | Triage + reply agent, LangSmith tracing, scored eval suite, regression gate | "Gate blocked a friendlier prompt: policy quoting 100% to 6%" |
| 2 | LLM-as-judge for tone and helpfulness, checked against hand labels | Judge-vs-human agreement % |
| 3 | Real retrieval (embeddings) with its own recall@k eval, separate from reply quality | Recall@k before/after |
| 4 | Feedback loop: turn low-scoring production traces into new dataset rows | Dataset size, regressions caught |
| 5 | Human approval for escalations via LangGraph interrupts + checkpointer | Time-to-handoff |
| 6 | Cost and latency budgets in the gate (tokens, p95) | $ per 1k tickets |
| 7 | Multi-turn conversations and conversation-level evals | Resolution rate |
| 8 | Prompt and model A/B write-up using LangSmith experiment comparison | Winner, with evidence |

Rules of the series: every week keeps `pytest` and `cx-gate` green, adds at least one test, and
updates the README's "Limits" section honestly.

## Week 1 LinkedIn post (edit the bracketed bits)

> I shipped a "friendlier" support prompt. It read great. My regression gate blocked it anyway.
>
> This week I built **cx-copilot**: a support triage-and-reply agent in LangGraph, traced with
> LangSmith, with an eval suite and a CI gate that blocks any prompt or model change that makes
> support worse.
>
> What the gate caught (31 labelled tickets, runs offline in under a second):
> - Friendlier prompt: quality 0.99 -> 0.84. Policy quoting fell from 100% to 6%, and every
>   first draft needed a rewrite. Blocked.
> - Cheaper model: overall quality only dipped 0.02, inside the tolerance. But escalation recall
>   went from 100% to 50%: it stopped escalating angry, urgent customers. Blocked.
>
> Lesson: an average hides the failure that matters. Gate each metric on its own.
>
> Design choices I liked: risky tickets never get a free-form LLM reply, a regex safety net
> overrides the model, and unparseable triage fails safe to a human.
>
> Code, tests and the gate: [repo link]
> Next week: an LLM judge, calibrated against human labels.
>
> #LangGraph #LangSmith #LLMEvals #CustomerExperience

Attach: a screenshot of the BLOCKED table (`make demo`) and one LangSmith trace of the graph
(run `cx-copilot --trace "..."` with your own key).

## Publishing checklist

1. `pytest -q` and `cx-gate` green.
2. README "Limits" updated.
3. `git tag v0.N && git push --tags`.
4. Post with one screenshot and one number.
