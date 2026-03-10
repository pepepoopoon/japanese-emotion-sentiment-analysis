from japanese_emotion_sentiment.demo_data import make_smoke_data
from japanese_emotion_sentiment.evaluate import evaluate
from japanese_emotion_sentiment.predict import predict
from japanese_emotion_sentiment.train import train


def test_end_to_end(tmp_path) -> None:
    data_path = tmp_path / "japanese.csv"
    artifact_dir = tmp_path / "artifacts"
    make_smoke_data().to_csv(data_path, index=False)

    metadata = train(data_path, artifact_dir, random_state=42)
    metrics = evaluate(artifact_dir / "model.joblib", artifact_dir / "test.csv")
    prediction = predict(artifact_dir / "model.joblib", "合格して本当にうれしい")

    assert len(metadata["emotion_thresholds"]) == 8
    assert 0.0 <= metrics["emotions"]["micro_f1"] <= 1.0
    assert 0.0 <= metrics["emotions"]["macro_f1"] <= 1.0
    assert 0.0 <= metrics["emotions"]["hamming_loss"] <= 1.0
    assert prediction["polarity"] == "positive"
    assert "joy" in prediction["emotions"]
    assert (artifact_dir / "split_manifest.csv").exists()
