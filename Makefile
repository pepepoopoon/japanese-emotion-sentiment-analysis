.PHONY: install lint test smoke train evaluate predict

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

smoke:
	japanese-generate-smoke --output data/smoke.csv

train:
	japanese-train --data data/smoke.csv --output-dir artifacts/smoke

evaluate:
	japanese-evaluate --model artifacts/smoke/model.joblib --data artifacts/smoke/test.csv --output artifacts/smoke/test_metrics.json

predict:
	japanese-predict --model artifacts/smoke/model.joblib --text "合格して本当にうれしい"
