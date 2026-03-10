"""CLI обучения моделей японского текста."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib

from .data import EMOTION_COLUMNS, load_csv, stratified_split
from .io import write_json
from .model import (
    build_emotion_model,
    build_polarity_model,
    build_vectorizer,
    evaluate_bundle,
    tune_thresholds,
)

LOGGER = logging.getLogger(__name__)


def train(data_path: str | Path, output_dir: str | Path, *, random_state: int = 42) -> dict:
    data = load_csv(data_path)
    train_frame, validation_frame, test_frame, manifest = stratified_split(
        data, random_state=random_state
    )
    vectorizer = build_vectorizer()
    train_features = vectorizer.fit_transform(train_frame["text"])
    validation_features = vectorizer.transform(validation_frame["text"])

    polarity_model = build_polarity_model(random_state=random_state)
    polarity_model.fit(train_features, train_frame["polarity"])
    emotion_model = build_emotion_model(random_state=random_state)
    emotion_model.fit(train_features, train_frame.loc[:, EMOTION_COLUMNS].to_numpy())
    validation_probabilities = emotion_model.predict_proba(validation_features)
    thresholds = tune_thresholds(
        validation_frame.loc[:, EMOTION_COLUMNS].to_numpy(), validation_probabilities
    )
    bundle = {
        "schema_version": 1,
        "random_state": random_state,
        "vectorizer": vectorizer,
        "polarity_model": polarity_model,
        "emotion_model": emotion_model,
        "emotion_columns": list(EMOTION_COLUMNS),
        "emotion_thresholds": thresholds.tolist(),
    }
    validation_metrics = evaluate_bundle(bundle, validation_frame, EMOTION_COLUMNS)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination / "model.joblib")
    manifest.to_csv(destination / "split_manifest.csv", index=False)
    validation_frame.to_csv(destination / "validation.csv", index=False)
    test_frame.to_csv(destination / "test.csv", index=False)
    metadata = {
        "schema_version": 1,
        "random_state": random_state,
        "split_strategy": "stratified by polarity",
        "split_rows": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "emotion_thresholds": dict(zip(EMOTION_COLUMNS, thresholds.tolist(), strict=True)),
        "validation_metrics": validation_metrics,
    }
    write_json(destination / "metadata.json", metadata)
    LOGGER.info("Артефакты сохранены в %s", destination)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train(args.data, args.output_dir, random_state=args.seed)


if __name__ == "__main__":
    main()
