# Your first real-model run

The offline numbers show the harness works. This is how to get numbers about real models.

1. **Install a free local model.** Install [Ollama](https://ollama.com), then pull two models to
   compare, for example `ollama pull llama3.2` and one larger model your machine can run.
2. **Smoke test one ticket.** If this works, the adapter, prompts and JSON parsing work.
   ```bash
   cx-copilot --model ollama:llama3.2 "I was charged twice for order #4471"
   ```
   Check the triage line is sensible and the reply quotes the policy figures.
3. **Compare the two models.**
   ```bash
   cx-compare --models ollama:<larger> ollama:llama3.2 --repeats 3 --cache .cx_cache
   ```
   About 90 calls per model per repeat, all local and free. Rerunning is instant because of `--cache`.
   Optional: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` and add `--trace` to see every run in
   LangSmith (free plan: 5,000 traces a month; 3 repeats of 2 models is about 190 traces).
4. **Try the triage fix on the same tickets.** Responses already in the cache are reused.
   ```bash
   cx-compare --models ollama:<larger> ollama:<larger>@triage_v2/rules \
                       ollama:llama3.2 ollama:llama3.2@triage_v2/rules --repeats 3 --cache .cx_cache
   ```
5. **Read `evals/compare.md`.** Look at the verdict line and the spread:
   - Moved by more than the spread, and the gate says BLOCKED: that is a real finding. Name the
     metric. This is the post.
   - PASS, with small moves: "the gate allowed the swap" is also an honest, useful result.
   - Lots of errors: fix the cause (key, model name, rate limit) and rerun. Do not report it.
6. **Look at the misses.** The "Where each one went wrong" section lists every missed escalation,
   over-escalation and wrong category, with the ticket text. Some will be real mistakes. Some will be scorer false alarms (for example "thirty days"
   instead of "30 days"). Say which in your write-up.
7. **Save baselines** per model with `cx-gate --model <spec> --baseline evals/baseline.<name>.json
   --update-baseline`, and commit them.
8. **Trace once.** `cx-copilot --trace --model ollama:llama3.2 "..."`, then screenshot the run in LangSmith.

## Rules for reporting it

- Do not change the scorers after seeing results to make a number look better. If you must, say
  you did and why.
- Report the number of repeats and the spread.
- The dataset is 31 tickets: one ticket is about 3 points. Do not oversell small gaps.
- Replace the stand-in numbers in the README and any message with the real ones, and delete the
  stand-in caveat only when the claim underneath it is real.
