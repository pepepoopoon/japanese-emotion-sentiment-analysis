"""Run deterministic Japanese sentiment and emotion experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from .data import EMOTION_COLUMNS, stratified_split, validate_frame
from .demo_data import make_smoke_data
from .io import write_json
from .model import (
    build_emotion_model,
    build_polarity_model,
    build_vectorizer,
    emotion_metrics,
    evaluate_bundle,
    tune_thresholds,
)

RESULT_SCHEMA_VERSION = 1


def baseline_metrics(
    train_features: object,
    train_frame: object,
    evaluation_features: object,
    evaluation_frame: object,
) -> dict[str, object]:
    """Evaluate prior polarity and prevalence emotion baselines."""
    polarity = DummyClassifier(strategy="prior")
    polarity.fit(train_features, train_frame["polarity"])
    polarity_predictions = polarity.predict(evaluation_features)
    emotion_prevalence = train_frame.loc[:, EMOTION_COLUMNS].to_numpy().mean(axis=0)
    emotion_probabilities = np.tile(emotion_prevalence, (len(evaluation_frame), 1))
    return {
        "rows": len(evaluation_frame),
        "polarity_macro_f1": float(
            f1_score(
                evaluation_frame["polarity"],
                polarity_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "emotions": emotion_metrics(
            evaluation_frame.loc[:, EMOTION_COLUMNS].to_numpy(),
            emotion_probabilities,
            np.full(len(EMOTION_COLUMNS), 0.5),
        ),
    }


def run_experiment(*, seed: int = 42) -> dict[str, object]:
    """Fit the classical model and evaluate deterministic validation/test splits."""
    frame = validate_frame(make_smoke_data())
    train, validation, test, _ = stratified_split(frame, random_state=seed)
    vectorizer = build_vectorizer()
    train_features = vectorizer.fit_transform(train["text"])
    validation_features = vectorizer.transform(validation["text"])

    polarity_model = build_polarity_model(random_state=seed)
    polarity_model.fit(train_features, train["polarity"])
    emotion_model = build_emotion_model(random_state=seed)
    emotion_model.fit(train_features, train.loc[:, EMOTION_COLUMNS].to_numpy())
    thresholds = tune_thresholds(
        validation.loc[:, EMOTION_COLUMNS].to_numpy(),
        emotion_model.predict_proba(validation_features),
    )
    bundle = {
        "vectorizer": vectorizer,
        "polarity_model": polarity_model,
        "emotion_model": emotion_model,
        "emotion_thresholds": thresholds,
    }
    validation_metrics = evaluate_bundle(bundle, validation, EMOTION_COLUMNS)
    test_metrics = evaluate_bundle(bundle, test, EMOTION_COLUMNS)
    validation_baseline = baseline_metrics(
        train_features,
        train,
        validation_features,
        validation,
    )
    test_baseline = baseline_metrics(
        train_features,
        train,
        vectorizer.transform(test["text"]),
        test,
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "config": {"seed": seed},
        "data": {
            "rows": len(frame),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "model": "char_tfidf_logistic_regression",
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "baseline_metrics": {"validation": validation_baseline, "test": test_baseline},
        "delta_vs_baseline": {
            "test_polarity_macro_f1": (
                test_metrics["polarity_macro_f1"] - test_baseline["polarity_macro_f1"]
            ),
            "test_emotion_micro_f1": (
                test_metrics["emotions"]["micro_f1"] - test_baseline["emotions"]["micro_f1"]
            ),
        },
        "emotion_thresholds": dict(zip(EMOTION_COLUMNS, thresholds.tolist(), strict=True)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    write_json(args.output, run_experiment(seed=args.seed))
    print(f"experiment result written to {args.output}")


if __name__ == "__main__":
    main()
