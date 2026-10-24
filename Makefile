.PHONY: install test gate demo

install:
	pip install -e ".[dev]"

test:
	pytest -q

gate:
	cx-gate

demo:
	-cx-gate --prompt reply_v3_friendly
	-cx-gate --model fake:lite
