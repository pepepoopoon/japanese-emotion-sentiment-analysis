"""Модели, пороги и метрики."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
        max_features=50_000,
    )


def build_polarity_model(*, random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_state,
    )


def build_emotion_model(*, random_state: int = 42) -> OneVsRestClassifier:
    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_state,
    )
    return OneVsRestClassifier(estimator)


def tune_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Подобрать независимый порог каждого класса по validation F1."""
    if y_true.shape != probabilities.shape:
        raise ValueError("Размерности меток и вероятностей не совпадают")
    candidates = np.linspace(0.2, 0.8, 25)
    thresholds: list[float] = []
    for column in range(y_true.shape[1]):
        scores = [
            f1_score(
                y_true[:, column],
                probabilities[:, column] >= threshold,
                zero_division=0,
            )
            for threshold in candidates
        ]
        best = max(
            range(len(candidates)),
            key=lambda index: (scores[index], -abs(candidates[index] - 0.5)),
        )
        thresholds.append(float(candidates[best]))
    return np.asarray(thresholds)


def emotion_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    predictions = (probabilities >= thresholds).astype(int)
    return {
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, predictions)),
    }


def evaluate_bundle(
    bundle: dict[str, Any],
    frame: Any,
    emotion_columns: tuple[str, ...],
) -> dict[str, Any]:
    features = bundle["vectorizer"].transform(frame["text"])
    polarity_predictions = bundle["polarity_model"].predict(features)
    emotion_probabilities = bundle["emotion_model"].predict_proba(features)
    polarity_macro_f1 = f1_score(
        frame["polarity"], polarity_predictions, average="macro", zero_division=0
    )
    metrics = emotion_metrics(
        frame.loc[:, emotion_columns].to_numpy(),
        emotion_probabilities,
        np.asarray(bundle["emotion_thresholds"]),
    )
    return {
        "rows": int(len(frame)),
        "polarity_macro_f1": float(polarity_macro_f1),
        "emotions": metrics,
    }
