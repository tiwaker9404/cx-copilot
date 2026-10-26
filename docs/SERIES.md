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

## Week 1 LinkedIn post (real results)

> My 7B model scored higher than the 3B overall, and still ignored a customer who needed a human.
>
> I built cx-copilot: a support agent in LangGraph, traced in LangSmith, with a regression gate that
> blocks prompt or model changes that make support worse. It runs on free local models, so the whole
> thing cost $0.
>
> 31 labelled tickets, 3 runs each:
> - qwen2.5 7B beat llama3.2 3B on overall quality and categories. But "Worst service ever...
>   I need it TODAY. This is ridiculous." got labelled frustrated, not angry, and was auto-answered in
>   2 of 3 runs.
> - llama3.2 went the other way: it called "I was charged twice" angry and urgent, and sent about 9
>   of every 31 tickets to a human for no reason.
>
> Two changes fixed the 7B: define every label with examples, and take the escalation decision away
> from the model. The model labels; code decides who goes to a human. Result: 0 missed, 0
> unnecessary escalations, quality 0.91 to 0.96.
>
> Averages hid the one failure that mattered. Gate each metric on its own.
>
> How are you deciding which of your agent's decisions stay with the model?
>
> #LangGraph #LangSmith #LLMEvals

**First comment:** "Code, eval data and the comparison: [repo link]"
**Attach:** a screenshot of the "Where each one went wrong" section of `evals/compare.md`, or the
LangSmith trace of `t21`.

## Publishing checklist

1. `pytest -q` and `cx-gate` green.
2. README "Limits" updated.
3. `git tag v0.N && git push --tags`.
4. Post with one screenshot and one number.
