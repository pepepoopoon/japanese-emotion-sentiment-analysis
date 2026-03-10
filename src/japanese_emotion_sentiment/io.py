"""Сериализация артефактов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .data import EMOTION_COLUMNS

MODEL_SCHEMA_VERSION = 1


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("Файл не содержит bundle-словарь модели")

    required = {
        "schema_version",
        "vectorizer",
        "polarity_model",
        "emotion_model",
        "emotion_columns",
        "emotion_thresholds",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"В bundle отсутствуют обязательные поля: {sorted(missing)}")
    if bundle["schema_version"] != MODEL_SCHEMA_VERSION:
        raise ValueError(
            "Неподдерживаемая schema_version bundle: "
            f"{bundle['schema_version']}; ожидается {MODEL_SCHEMA_VERSION}"
        )

    emotion_columns = bundle["emotion_columns"]
    if not isinstance(emotion_columns, (list, tuple)) or tuple(emotion_columns) != EMOTION_COLUMNS:
        raise ValueError("Порядок emotion_columns не соответствует schema_version bundle")
    try:
        thresholds = np.asarray(bundle["emotion_thresholds"], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("emotion_thresholds должны быть числовыми") from error
    if thresholds.shape != (len(EMOTION_COLUMNS),):
        raise ValueError("Число emotion_thresholds не совпадает с emotion_columns")
    if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("emotion_thresholds должны быть конечными значениями от 0 до 1")
    return bundle
