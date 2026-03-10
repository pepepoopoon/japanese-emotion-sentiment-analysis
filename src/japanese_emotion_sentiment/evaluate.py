"""CLI оценки сохранённых моделей."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import EMOTION_COLUMNS, load_csv
from .io import load_bundle, write_json
from .model import evaluate_bundle


def evaluate(model_path: str | Path, data_path: str | Path) -> dict:
    bundle = load_bundle(model_path)
    return evaluate_bundle(bundle, load_csv(data_path), EMOTION_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    metrics = evaluate(args.model, args.data)
    if args.output:
        write_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
