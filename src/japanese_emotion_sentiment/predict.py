"""CLI инференса полярности и эмоций."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import normalize_text
from .io import load_bundle


def predict(model_path: str | Path, text: str) -> dict[str, Any]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        raise ValueError("Текст для предсказания не может быть пустым")
    bundle = load_bundle(model_path)
    features = bundle["vectorizer"].transform([normalized_text])
    polarity_probabilities = bundle["polarity_model"].predict_proba(features)[0]
    polarity_classes = bundle["polarity_model"].classes_
    polarity_index = int(polarity_probabilities.argmax())
    emotion_probabilities = bundle["emotion_model"].predict_proba(features)[0]
    emotions = []
    scores: dict[str, float] = {}
    for name, probability, threshold in zip(
        bundle["emotion_columns"],
        emotion_probabilities,
        bundle["emotion_thresholds"],
        strict=True,
    ):
        scores[name] = float(probability)
        if probability >= threshold:
            emotions.append(name.removeprefix("emotion_"))
    return {
        "polarity": str(polarity_classes[polarity_index]),
        "polarity_confidence": float(polarity_probabilities[polarity_index]),
        "emotions": emotions,
        "emotion_probabilities": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.model, args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
