"""Сериализация артефактов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    required = {"vectorizer", "polarity_model", "emotion_model", "emotion_thresholds"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Файл не содержит поддерживаемый bundle модели")
    return bundle
