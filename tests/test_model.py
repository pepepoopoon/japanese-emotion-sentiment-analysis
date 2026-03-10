import numpy as np

from japanese_emotion_sentiment.model import emotion_metrics, tune_thresholds


def test_thresholds_and_multilabel_metrics() -> None:
    labels = np.asarray([[1, 0], [1, 1], [0, 1], [0, 0]])
    probabilities = np.asarray([[0.9, 0.1], [0.8, 0.7], [0.2, 0.8], [0.1, 0.2]])

    thresholds = tune_thresholds(labels, probabilities)
    metrics = emotion_metrics(labels, probabilities, thresholds)

    assert thresholds.shape == (2,)
    assert metrics == {"micro_f1": 1.0, "macro_f1": 1.0, "hamming_loss": 0.0}
