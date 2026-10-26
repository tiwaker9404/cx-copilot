## cx-copilot regression gate: PASS

Candidate: `reply_v4_grounded` on `ollama:qwen2.5:7b`  |  Baseline: `reply_v1_baseline` on `ollama:qwen2.5:7b`

| Metric | Baseline | Candidate | Change | Allowed drop | Status |
|---|---|---|---|---|---|
| quality | 0.900 | 0.918 | +0.018 | 0.02 | ok |
| category_accuracy | 0.832 | 0.832 | +0.000 | 0.05 | ok |
| policy_grounded | 0.799 | 0.784 | -0.015 | 0.05 | ok |
| empathy | 0.833 | 0.896 | +0.062 | 0.05 | ok |
| actionable | 0.808 | 0.941 | +0.133 | 0.05 | ok |
| length_ok | 1.000 | 1.000 | +0.000 | 0.10 | ok |
| first_try_clean | 0.913 | 0.952 | +0.038 | 0.10 | ok |
| escalation_correct | 0.952 | 0.976 | +0.024 | 0.05 | ok |
| escalation_recall | 1.000 | 1.000 | +0.000 | 0.00 | ok |
| safety | 1.000 | 1.000 | +0.000 | 0.00 | ok |
| no_invented_commitments | 1.000 | 1.000 | +0.000 | 0.00 | ok |
| no_repeat_questions | 1.000 | 1.000 | +0.000 | 0.05 | ok |
| draft_commitment_free | 0.933 | 0.952 | +0.019 | 0.05 | ok |
| followup_not_repeated | 1.000 | 1.000 | +0.000 | 0.10 | ok |
| retrieval_recall | 0.978 | 0.979 | +0.000 | 0.05 | ok |

Biggest per-ticket drops: t07 (0.90 -> 0.65), m04 (0.86 -> 0.62), m10 (0.86 -> 0.62), t10 (1.00 -> 0.78), t47 (1.00 -> 0.78)

| Budget (per ticket) | Limit | Actual | Status |
|---|---|---|---|
| mean_calls_per_ticket | 2.50 | 1.87 | ok |
| mean_tokens_per_ticket | 2,500 | 1,073 | ok |
| p95_model_ms_per_ticket | 60,000 | 14,806 | ok |

QA guardrail caught in first drafts: invented menu path x1, invented timeline x6
