from collections.abc import Callable

import joblib
import pytest

from japanese_emotion_sentiment.data import EMOTION_COLUMNS
from japanese_emotion_sentiment.io import MODEL_SCHEMA_VERSION, load_bundle


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda bundle: bundle.update(schema_version=2), "Неподдерживаемая schema_version"),
        (
            lambda bundle: bundle.update(emotion_columns=list(reversed(EMOTION_COLUMNS))),
            "Порядок emotion_columns",
        ),
        (
            lambda bundle: bundle.update(emotion_thresholds=[0.5]),
            "Число emotion_thresholds",
        ),
    ],
)
def test_load_bundle_rejects_incompatible_emotion_contract(
    tmp_path,
    mutate: Callable[[dict], None],
    message: str,
) -> None:
    bundle = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "vectorizer": object(),
        "polarity_model": object(),
        "emotion_model": object(),
        "emotion_columns": list(EMOTION_COLUMNS),
        "emotion_thresholds": [0.5] * len(EMOTION_COLUMNS),
    }
    mutate(bundle)
    model_path = tmp_path / "model.joblib"
    joblib.dump(bundle, model_path)

    with pytest.raises(ValueError, match=message):
        load_bundle(model_path)
