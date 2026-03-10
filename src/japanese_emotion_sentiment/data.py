"""Контракт и разбиение данных."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

EMOTION_COLUMNS = (
    "emotion_joy",
    "emotion_sadness",
    "emotion_anticipation",
    "emotion_surprise",
    "emotion_anger",
    "emotion_fear",
    "emotion_disgust",
    "emotion_trust",
)
POLARITIES = frozenset({"negative", "neutral", "positive"})
REQUIRED_COLUMNS = ("text_id", "text", "polarity", *EMOTION_COLUMNS)


def normalize_text(text: str) -> str:
    """Привести совместимые Unicode-варианты к единому представлению."""
    return unicodedata.normalize("NFKC", text).strip()


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Проверить схему разметки полярности и эмоций."""
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные столбцы: {sorted(missing)}")
    data = frame.copy()
    for column in ("text_id", "text", "polarity"):
        if data[column].isna().any():
            raise ValueError(f"Столбец {column!r} содержит пропуски")
        data[column] = data[column].astype(str).str.strip()
        if column == "text":
            data[column] = data[column].map(normalize_text)
        if data[column].eq("").any():
            raise ValueError(f"Столбец {column!r} содержит пустые значения")
    if data["text_id"].duplicated().any():
        raise ValueError("text_id должен быть уникальным")
    if data["text"].duplicated().any():
        raise ValueError("Повторяющиеся тексты необходимо удалить до split")
    unknown = set(data["polarity"]).difference(POLARITIES)
    if unknown:
        raise ValueError(f"Неизвестные значения polarity: {sorted(unknown)}")
    for column in EMOTION_COLUMNS:
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise ValueError(f"{column} должен содержать только 0 или 1")
        data[column] = numeric.astype("int8")
    return data


def load_csv(path: str | Path) -> pd.DataFrame:
    return validate_frame(pd.read_csv(path))


def stratified_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделить строки, сохраняя распределение полярности."""
    data = validate_frame(frame)
    counts = data["polarity"].value_counts()
    scarce = counts[counts < 5]
    if not scarce.empty:
        raise ValueError(
            f"Для трёхчастного split нужно минимум 5 строк на polarity: {scarce.to_dict()}"
        )
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Доли validation и test должны быть положительными")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Сумма долей validation и test должна быть меньше 1")
    train_validation, test = train_test_split(
        data,
        test_size=test_fraction,
        random_state=random_state,
        stratify=data["polarity"],
    )
    relative_validation = validation_fraction / (1 - test_fraction)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation,
        random_state=random_state,
        stratify=train_validation["polarity"],
    )
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    constant_train_labels = [column for column in EMOTION_COLUMNS if train[column].nunique() < 2]
    if constant_train_labels:
        raise ValueError(
            "В train отсутствуют оба значения для emotion-меток: "
            f"{constant_train_labels}. Добавьте данные или измените seed."
        )
    manifest = pd.concat(
        [
            train[["text_id", "polarity"]].assign(split="train"),
            validation[["text_id", "polarity"]].assign(split="validation"),
            test[["text_id", "polarity"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    return train, validation, test, manifest
