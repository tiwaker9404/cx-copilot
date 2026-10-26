# The real-model run

The offline numbers show the harness works. This is how to get numbers about real models, for free,
on your own machine.

## One command

```bash
ollama pull qwen2.5:7b
ollama pull llama3.2
ollama pull nomic-embed-text
pip install -e ".[dev]"
bash scripts/full_eval.sh
```

What it does, in order (all model calls are cached in `.cx_cache`, so stopping and rerunning is safe
and finished steps cost nothing):

| Step | Output | Time on an Apple-silicon laptop |
|---|---|---|
| Tests | pass/fail | 15 s |
| Retrieval: keyword, BM25, embeddings, hybrid | `evals/retrieval.md` | 1 min |
| Judge calibration on 40 human labels, both models | `evals/judge_calibration*.md` | 5 min |
| Compare qwen + v1, qwen + v4, llama + v4, with the judge | `evals/compare_v03.md` | about 1 h |
| Gate: qwen + v4 against a qwen + v1 baseline, with budgets | `evals/gate_v4.md` | seconds (cache) |
| Bundle | `cx-results.zip` | - |

Override the models with environment variables, for example
`MODEL=ollama:qwen2.5:7b JUDGE=ollama:llama3.2 bash scripts/full_eval.sh`. `EXPERIMENT=1` also logs
the comparison as LangSmith experiments (needs `LANGSMITH_API_KEY`; about 400 traces of the free
plan's 5,000 a month).

Keep the laptop plugged in and awake (`caffeinate -i bash scripts/full_eval.sh` on a Mac).

## Smoke test first (30 seconds)

```bash
cx-copilot --model ollama:qwen2.5:7b --prompt reply_v4_grounded "I was charged twice for order #4471"
```

The triage line should say billing, and the reply should quote "3-5 business days" and not ask for
the order number. If the first draft was sent back, the CLI prints why.

## Reading the results

- **`compare_v03.md`, "What the QA guardrail caught"**: how often each setup's first drafts
  invented a timeline, an offer or a menu path, or asked again for something the customer gave.
  `draft_commitment_free` is the same thing as a rate. This is the made-up-promises finding.
- **`no_invented_commitments`** should be 1.000 everywhere: that is the guardrail working. Anything
  lower is a bug to report, not a model result.
- **Judge rows** only mean something for criteria the calibration report marks "trusted".
- **"Where each one went wrong"** lists every missed escalation, over-escalation and wrong
  category with the ticket text. Some are real mistakes, some are label arguments; say which.
- **`gate_v4.md`**: whether swapping v1 for v4 passes every per-metric limit and the budgets.

## Rules for reporting it

- Do not change scorers or labels after seeing results to make a number look better. If you must,
  say you did and why.
- One repeat per setup here (time). Treat differences of a couple of points as noise; 125 tickets
  means one ticket is under 1 point, but one run is still one run.
- Name the machine, the models and the date.
