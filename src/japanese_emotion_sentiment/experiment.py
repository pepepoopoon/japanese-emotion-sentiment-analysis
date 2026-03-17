"""Run deterministic Japanese sentiment and emotion experiments."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion

from .data import EMOTION_COLUMNS, normalize_text, stratified_split, validate_frame
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
FEATURE_MODES = ("char", "word", "char_word")
STRESS_MODES = ("none", "unicode", "punctuation")
PUNCTUATION_STYLES = ("remove", "repeat", "emoji", "mixed")


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


def per_emotion_diagnostics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, object]:
    """Report support and thresholded quality for every emotion."""
    predictions = probabilities >= thresholds
    diagnostics: dict[str, object] = {}
    for index, emotion in enumerate(EMOTION_COLUMNS):
        diagnostics[emotion] = {
            "support": int(truth[:, index].sum()),
            "prevalence": float(truth[:, index].mean()),
            "predicted_positive": int(predictions[:, index].sum()),
            "mean_probability": float(probabilities[:, index].mean()),
            "threshold": float(thresholds[index]),
            "precision": float(
                precision_score(truth[:, index], predictions[:, index], zero_division=0)
            ),
            "recall": float(recall_score(truth[:, index], predictions[:, index], zero_division=0)),
            "f1": float(f1_score(truth[:, index], predictions[:, index], zero_division=0)),
        }
    return diagnostics


def build_experiment_vectorizer(feature_mode: str) -> object:
    """Build comparable character, word, or combined TF-IDF features."""
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unsupported feature_mode: {feature_mode}")
    if feature_mode == "char":
        return build_vectorizer()
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=50_000,
    )
    if feature_mode == "word":
        return word
    return FeatureUnion([("char", build_vectorizer()), ("word", word)])


def unicode_variant(text: str) -> str:
    """Replace ASCII code points with NFKC-equivalent full-width forms."""
    return "".join(
        chr(ord(character) + 0xFEE0) if "!" <= character <= "~" else character for character in text
    )


def punctuation_variant(text: str, style: str) -> str:
    """Apply one deterministic punctuation perturbation."""
    if style not in PUNCTUATION_STYLES:
        raise ValueError(f"unsupported punctuation_style: {style}")
    if style == "remove":
        return text.translate(str.maketrans("", "", "。！？!?、,."))
    if style == "repeat":
        return text.replace("。", "！！！").replace("！", "！！！")
    if style == "emoji":
        return f"{text}🙂"
    return f"【{text.translate(str.maketrans('', '', '。、'))}】!"


def stress_frame(
    frame: pd.DataFrame,
    *,
    fraction: float,
    seed: int,
    transform: object,
) -> pd.DataFrame:
    """Apply a deterministic text transform to a fraction of held-out rows."""
    if not 0 <= fraction <= 1:
        raise ValueError("stress_fraction must be in [0, 1]")
    result = frame.copy()
    if not fraction:
        return result
    count = min(len(result), max(1, math.ceil(len(result) * fraction)))
    positions = np.random.default_rng(seed).choice(len(result), size=count, replace=False)
    indices = result.index[positions]
    result.loc[indices, "text"] = result.loc[indices, "text"].map(transform)
    return result


def run_experiment(
    *,
    seed: int = 42,
    feature_mode: str = "char",
    stress_mode: str = "none",
    stress_fraction: float = 0.0,
    punctuation_style: str = "mixed",
) -> dict[str, object]:
    """Fit the classical model and evaluate deterministic validation/test splits."""
    frame = validate_frame(make_smoke_data())
    train, validation, test, _ = stratified_split(frame, random_state=seed)
    if stress_mode not in STRESS_MODES:
        raise ValueError(f"unsupported stress_mode: {stress_mode}")
    if not 0 <= stress_fraction <= 1:
        raise ValueError("stress_fraction must be in [0, 1]")
    if punctuation_style not in PUNCTUATION_STYLES:
        raise ValueError(f"unsupported punctuation_style: {punctuation_style}")
    vectorizer = build_experiment_vectorizer(feature_mode)
    train_features = vectorizer.fit_transform(train["text"])
    validation_features = vectorizer.transform(validation["text"])

    polarity_model = build_polarity_model(random_state=seed)
    polarity_model.fit(train_features, train["polarity"])
    emotion_model = build_emotion_model(random_state=seed)
    emotion_model.fit(train_features, train.loc[:, EMOTION_COLUMNS].to_numpy())
    validation_probabilities = emotion_model.predict_proba(validation_features)
    thresholds = tune_thresholds(
        validation.loc[:, EMOTION_COLUMNS].to_numpy(), validation_probabilities
    )
    bundle = {
        "vectorizer": vectorizer,
        "polarity_model": polarity_model,
        "emotion_model": emotion_model,
        "emotion_thresholds": thresholds,
    }
    stress_diagnostics: dict[str, object] = {
        "mode": stress_mode,
        "fraction": stress_fraction,
        "selected_rows": 0,
    }
    if stress_mode == "unicode":
        original_test = test.copy()
        raw_test = stress_frame(
            test,
            fraction=stress_fraction,
            seed=seed + 1,
            transform=unicode_variant,
        )
        normalized_test = raw_test.copy()
        normalized_test["text"] = normalized_test["text"].map(normalize_text)
        selected_rows = int(raw_test["text"].ne(original_test["text"]).sum())
        stress_diagnostics = {
            "mode": stress_mode,
            "fraction": stress_fraction,
            "selected_rows": selected_rows,
            "normalized_recovered_rows": int(
                normalized_test["text"].eq(original_test["text"]).sum()
            ),
            "raw_metrics": evaluate_bundle(bundle, raw_test, EMOTION_COLUMNS),
            "normalized_metrics": evaluate_bundle(bundle, normalized_test, EMOTION_COLUMNS),
        }
        test = normalized_test
    elif stress_mode == "punctuation":
        original_test = test.copy()
        stressed_test = stress_frame(
            test,
            fraction=stress_fraction,
            seed=seed + 1,
            transform=lambda text: punctuation_variant(text, punctuation_style),
        )
        stressed_test["text"] = stressed_test["text"].map(normalize_text)
        stress_diagnostics = {
            "mode": stress_mode,
            "style": punctuation_style,
            "fraction": stress_fraction,
            "selected_rows": int(stressed_test["text"].ne(original_test["text"]).sum()),
            "stressed_metrics": evaluate_bundle(bundle, stressed_test, EMOTION_COLUMNS),
        }
        test = stressed_test
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
    test_probabilities = emotion_model.predict_proba(vectorizer.transform(test["text"]))
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "config": {
            "feature_mode": feature_mode,
            "seed": seed,
            "stress_fraction": stress_fraction,
            "stress_mode": stress_mode,
            "punctuation_style": punctuation_style,
        },
        "data": {
            "rows": len(frame),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "model": f"{feature_mode}_tfidf_logistic_regression",
        "feature_count": int(train_features.shape[1]),
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
        "per_emotion": {
            "validation": per_emotion_diagnostics(
                validation.loc[:, EMOTION_COLUMNS].to_numpy(),
                validation_probabilities,
                thresholds,
            ),
            "test": per_emotion_diagnostics(
                test.loc[:, EMOTION_COLUMNS].to_numpy(),
                test_probabilities,
                thresholds,
            ),
        },
        "stress_diagnostics": stress_diagnostics,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="char")
    parser.add_argument("--stress-mode", choices=STRESS_MODES, default="none")
    parser.add_argument("--stress-fraction", type=float, default=0.0)
    parser.add_argument("--punctuation-style", choices=PUNCTUATION_STYLES, default="mixed")
    args = parser.parse_args(argv)
    write_json(
        args.output,
        run_experiment(
            seed=args.seed,
            feature_mode=args.feature_mode,
            stress_mode=args.stress_mode,
            stress_fraction=args.stress_fraction,
            punctuation_style=args.punctuation_style,
        ),
    )
    print(f"experiment result written to {args.output}")


if __name__ == "__main__":
    main()
