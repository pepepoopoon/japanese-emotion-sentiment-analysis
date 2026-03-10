import pandas as pd
import pytest

from japanese_emotion_sentiment.data import EMOTION_COLUMNS, stratified_split, validate_frame
from japanese_emotion_sentiment.demo_data import make_smoke_data


def test_validation_rejects_non_binary_emotion() -> None:
    frame = make_smoke_data()
    frame.loc[0, EMOTION_COLUMNS[0]] = 2

    with pytest.raises(ValueError, match="только 0 или 1"):
        validate_frame(frame)


def test_validation_rejects_nfkc_equivalent_duplicate_texts() -> None:
    frame = make_smoke_data()
    frame.loc[0, "text"] = "テストＡＢＣ１２３"
    frame.loc[1, "text"] = "テストABC123"

    with pytest.raises(ValueError, match="Повторяющиеся тексты"):
        validate_frame(frame)


def test_stratified_split_is_disjoint_and_deterministic() -> None:
    first = stratified_split(make_smoke_data(), random_state=7)
    second = stratified_split(make_smoke_data(), random_state=7)
    train, validation, test, manifest = first

    assert set(train["text_id"]).isdisjoint(validation["text_id"])
    assert set(train["text_id"]).isdisjoint(test["text_id"])
    assert set(validation["text_id"]).isdisjoint(test["text_id"])
    assert all(part["polarity"].nunique() == 3 for part in (train, validation, test))
    pd.testing.assert_frame_equal(manifest, second[3])
