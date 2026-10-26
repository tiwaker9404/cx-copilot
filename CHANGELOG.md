# Changelog

## 0.3.0

- **Commitment guardrail.** QA now rejects drafts that invent timelines, refunds, credits,
  discounts, fee waivers, shipping upgrades, escalation or follow-up promises, or menu paths that
  the policy notes do not contain, and drafts that ask for an order number or email the customer
  already gave. New prompt `reply_v4_grounded`. New metrics `draft_commitment_free` (model
  behaviour) and `no_invented_commitments` (what was sent; zero tolerance in the gate).
- **Conversations.** Tickets take earlier messages (`history`). The thread is redacted, used for
  triage, retrieval and the reply; the third customer message escalates (`repeat_contact`).
  Metrics `no_repeat_questions`, `followup_not_repeated`.
- **Human review.** `CX_REVIEW_MODE=risky|all` pauses replies with LangGraph `interrupt()` and a
  SQLite checkpointer; `GET /v1/reviews`, `POST /v1/tickets/{id}/review` (approve / edit / reject).
  Survives restarts; exactly-once under concurrent reviewers.
- **Dataset 31 -> 125 tickets**, 20 multi-turn, every category and all 30 policies, labelled
  `relevant_policies`. Knowledge base 11 -> 30 policies.
- **Retrieval.** BM25, embeddings (Ollama `nomic-embed-text`) and a hybrid (new default) next to
  the original keyword matcher, which now matches whole words. `cx-retrieval` reports recall@k,
  MRR, precision and no-policy noise.
- **LLM judge.** `cx-judge` calibrates a judge against 40 human-labelled replies (agreement,
  Cohen's kappa, every disagreement). `--judge` on `cx-gate` / `cx-compare` adds judge_* metrics.
- **Cost and latency.** Token counts and latency are recorded per call (and kept in the cache);
  reports show calls, tokens and p50/p95 model time per ticket; `evals/budgets.json` is enforced by
  the gate.
- **Feedback loop.** `cx-harvest` turns review edits, rejections and low ratings into candidate
  eval cases.
- `cx-compare` entries can set the reply prompt (`MODEL+REPLY_PROMPT`), show what QA caught in first
  drafts, and log LangSmith experiments (`--experiment`; dataset synced under a content hash).
- The baseline records a dataset fingerprint; the gate refuses to compare scores from different
  ticket sets. Root runs are tagged with triage prompt, escalation mode and retrieval method.
- `scripts/full_eval.sh`: the whole real-model evaluation in one resumable command.
- From the real run (qwen2.5:7b, llama3.2, 125 tickets): `reply_v4_grounded` is the default;
  judge scores are reported but never gated; embeddings stay optional. Auditing every flagged
  draft found four false-alarm patterns in the commitment rules (arithmetic on backed figures,
  quoted or partial menu paths, forwarding a compliment), now fixed and tested, and a retrieval gap
  ("late" orders) that caused most invented timelines. Reports keep the flagged first draft.

## 0.2.0

- HTTP service (FastAPI): auth, rate limits, PII redaction, idempotency, fail-safe escalation,
  LangSmith feedback, Docker.
- `triage_v2` + rule-based escalation, chosen from real runs on qwen2.5:7b and llama3.2.
- `cx-compare`, response cache, Ollama and OpenAI-compatible adapters.

## 0.1.0

- LangGraph agent, offline stand-in model, scorers, regression gate, CI.
