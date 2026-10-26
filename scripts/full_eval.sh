#!/usr/bin/env bash
# Full evaluation on free local models (Ollama). About 1-1.5 hours on an Apple-silicon laptop.
#
#   bash scripts/full_eval.sh
#
# Safe to stop (Ctrl-C) and rerun: every model call is cached in .cx_cache, so a rerun picks
# up where it stopped and finished steps cost nothing. Results land in evals/ and are zipped
# into cx-results.zip.
#
# Override any of these:  MODEL=ollama:qwen2.5:7b SMALL=ollama:llama3.2 JUDGE=... bash scripts/full_eval.sh
# Add EXPERIMENT=1 to also log the comparison as LangSmith experiments (needs LANGSMITH_API_KEY;
# about 400 traces of the free plan's 5,000 a month).

set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-ollama:qwen2.5:7b}"
SMALL="${SMALL:-ollama:llama3.2}"
JUDGE="${JUDGE:-$MODEL}"
EMBED="${EMBED:-ollama:nomic-embed-text}"
CACHE="${CACHE:-.cx_cache}"
OLLAMA="${OLLAMA_HOST_URL:-http://localhost:11434}"
LOG=evals/full_eval.log
mkdir -p evals
: > "$LOG"

step() { printf '\n=== %s  [%s]\n' "$1" "$(date +%H:%M:%S)" | tee -a "$LOG"; }
run() { "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
need_model() {
  local name="${1#ollama:}"
  if ! curl -fsS "$OLLAMA/api/tags" | grep -Eq "\"name\": ?\"$name"; then
    echo "Missing Ollama model: $name. Run:  ollama pull $name" | tee -a "$LOG"
    return 1
  fi
}

step "Preflight"
if ! curl -fsS "$OLLAMA/api/tags" >/dev/null; then
  echo "Ollama is not running at $OLLAMA. Start the Ollama app (or: ollama serve) and rerun." | tee -a "$LOG"
  exit 2
fi
missing=0
for m in "$MODEL" "$SMALL" "$JUDGE"; do need_model "$m" || missing=1; done
[ "$missing" = 1 ] && exit 2
HAVE_EMBED=1
need_model "$EMBED" || { HAVE_EMBED=0; echo "(embedding retrieval will be skipped)" | tee -a "$LOG"; }
command -v cx-gate >/dev/null || { echo 'cx-copilot is not installed: pip install -e ".[dev]"' | tee -a "$LOG"; exit 2; }
echo "agent=$MODEL  small=$SMALL  judge=$JUDGE  embed=$([ $HAVE_EMBED = 1 ] && echo "$EMBED" || echo none)" | tee -a "$LOG"

step "1/6 Tests (offline, ~15 s)"
if python -m pytest -q -p no:warnings >> "$LOG" 2>&1; then
  grep -E "passed|failed" "$LOG" | tail -1
else
  echo "Tests failed (details in $LOG); fix before spending model time." | tee -a "$LOG"; exit 1
fi

step "2/6 Retrieval eval (~1 min)"
if [ "$HAVE_EMBED" = 1 ]; then
  run cx-retrieval --embed "$EMBED" --cache "$CACHE"
else
  run cx-retrieval
fi

step "3/6 Judge calibration against 40 human labels (~5 min)"
run cx-judge --model "$JUDGE" --cache "$CACHE" --out evals/judge_calibration
if [ "$SMALL" != "$JUDGE" ]; then
  run cx-judge --model "$SMALL" --cache "$CACHE" --out evals/judge_calibration_small
fi

step "4/6 Compare v1 vs v4 on $MODEL, plus v4 on $SMALL (the long one, ~1 h)"
extra=()
[ "${EXPERIMENT:-0}" = 1 ] && extra+=(--experiment)
run cx-compare --models "$MODEL+reply_v1_baseline" "$MODEL+reply_v4_grounded" "$SMALL+reply_v4_grounded" \
  --judge "$JUDGE" --cache "$CACHE" --progress --out evals/compare_v03 ${extra[@]+"${extra[@]}"}

step "5/6 Regression gate on the real model (replays the cache, ~10 s)"
tag="$(echo "${MODEL#ollama:}" | tr ':.' '__')"
run cx-gate --model "$MODEL" --prompt reply_v1_baseline --cache "$CACHE" --baseline "evals/baseline.$tag.json" --update-baseline --no-budgets
run cx-gate --model "$MODEL" --prompt reply_v4_grounded --cache "$CACHE" \
  --baseline "evals/baseline.$tag.json" --budgets evals/budgets.json > evals/gate_v4.md
echo "gate exit code: $? (0 pass, 1 blocked)" | tee -a "$LOG"
cat evals/gate_v4.md | tee -a "$LOG"

step "6/6 Bundle results"
rm -f cx-results.zip
zip -q cx-results.zip evals/*.md evals/*.json evals/full_eval.log
echo "Done. Send cx-results.zip (no keys or ticket data beyond the synthetic set are in it)." | tee -a "$LOG"
