# Operations

## Configuration

All settings are environment variables (see `.env.example`). The service validates them at startup
and refuses to boot on a bad value, so a typo never reaches a customer.

| Variable | Default | Notes |
|---|---|---|
| `CX_ENV` | `dev` | `prod` requires `CX_API_KEYS`, forbids the fake model, hides `/docs` |
| `CX_API_KEYS` | empty | comma-separated, 16+ characters each |
| `CX_MODEL` | `fake` | `ollama:<model>` is the free option |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | `http://ollama:11434/v1` inside compose |
| `CX_DB_PATH` | `data/cx.db` | SQLite file |
| `CX_KB_PATH` | built-in | JSON list of `{id, category, keywords, text}` |
| `CX_REPLY_PROMPT` | `reply_v4_grounded` | must exist in `src/cx_copilot/prompts/` |
| `CX_TRIAGE_PROMPT` | `triage_v2` | `triage_v1` is the original, kept for comparison |
| `CX_ESCALATION` | `rules` | the model labels, code decides who goes to a human; `model` trusts the model's own flag |
| `CX_REVIEW_MODE` | `off` | `risky` holds replies for a person when the customer is upset, the ticket is urgent, it is a follow-up, no policy matched or the first draft failed QA; `all` holds every auto-reply |
| `CX_CHECKPOINT_PATH` | `data/checkpoints.db` | where paused reviews are saved (`/data/checkpoints.db` in Docker) |
| `CX_RETRIEVAL` | `hybrid` | `keyword`, `bm25`, `embed` or `hybrid` |
| `CX_EMBED_MODEL` | empty | e.g. `ollama:nomic-embed-text`; adds embeddings to `hybrid`, required for `embed` |
| `CX_MAX_HISTORY` | `20` | earlier messages accepted per ticket |
| `CX_MAX_CONCURRENCY` | `4` | parallel model calls; 1-2 for a laptop running Ollama |
| `CX_QUEUE_TIMEOUT_S` | `30` | wait for a free slot, then `503` |
| `CX_RATE_LIMIT_PER_MIN` | `60` | per API key |
| `CX_MAX_TICKET_CHARS` | `4000` | longer tickets get `413` |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | off | tracing only runs when both flag and key are set |

## Staying on the free tier

- LangSmith Developer plan: one seat, 5,000 base traces a month, 14-day retention. One ticket is one
  trace. Above that it is pay-as-you-go, so watch the usage page or leave tracing off for bulk runs.
- Feedback is sent with `extend_trace_retention=False`, so it does not upgrade traces.
- `cx-compare` and `cx-gate` do not trace unless you pass `--trace`.
- Ollama runs on your hardware: no per-call cost. Expect a few seconds per ticket on a laptop.

## Changing the prompt or model safely

1. Add a new prompt file; never edit the live one in place.
2. `cx-compare --models <current> <candidate>` or `cx-gate --prompt <new> --model <spec> --baseline ...`.
3. Only if it passes: set `CX_REPLY_PROMPT` / `CX_MODEL` and restart.

## Human review

1. `GET /v1/reviews` lists held replies, oldest first, each with a `review_reason`.
2. `POST /v1/tickets/{id}/review` with `approve`, `edit` (+ `reply`) or `reject` (+ `note`).
   The response is the final result; only then is the reply sendable.
3. `409` means someone else already reviewed it, or it was never held.

Held tickets live in two places: the result in `CX_DB_PATH` (status `pending_review`) and the paused
graph run in `CX_CHECKPOINT_PATH`. Back up both together. A finished review deletes its checkpoint.
If a review crashes mid-resume, the ticket goes back to `pending_review` and can be retried.

`GET /v1/stats` shows `pending_review` and `review_edit_rate`. A rising edit rate is the earliest
signal that a prompt or model got worse in production.

## Turning reviews and ratings into tests

```bash
cx-harvest --db data/cx.db --out evals/candidates.jsonl
```

Fix the `gold` labels of each candidate (they start as the model's guesses), delete the helper
fields, append the rows to `src/cx_copilot/data/tickets.jsonl`, and rebuild the baseline with
`cx-gate --update-baseline`. The dataset fingerprint in the baseline changes with it.

## Day-to-day

- **Human queue:** `GET /v1/escalations`. `escalation_reason` says why: `safety_rule`, `triage`,
  `repeat_contact`, `qa` (two drafts failed the checks), `review_rejected`, `model_failure`.
  `status: "degraded"` means the model failed, not the customer.
- **Cost:** every result has `tokens` and `model_calls`. Budgets for evals are in `evals/budgets.json`.
- **Health:** `GET /v1/stats`. A rising `degraded` count means the model server is struggling.
- **Find a trace:** every result has a `run_id`; it is the LangSmith run id.
- **Rotate an API key:** add the new key to `CX_API_KEYS`, restart, move clients over, remove the
  old key, restart.
- **Backup:** `sqlite3 data/cx.db ".backup backup.db"` (and the same for `checkpoints.db`) works
  while the service is running.
- **Logs:** one JSON object per line on stdout, with `request_id`. Ticket text is never logged.
