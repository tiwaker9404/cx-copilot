.PHONY: install test gate demo retrieval judge full-eval harvest

install:
	pip install -e ".[dev]"

test:
	pytest -q

gate:
	cx-gate

demo:
	-cx-gate --prompt reply_v3_friendly
	-cx-gate --model fake:lite

retrieval:
	cx-retrieval

judge:
	cx-judge

full-eval:
	bash scripts/full_eval.sh

harvest:
	cx-harvest --db data/cx.db --out evals/candidates.jsonl
