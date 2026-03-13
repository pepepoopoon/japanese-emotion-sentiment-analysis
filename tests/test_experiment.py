from __future__ import annotations

import json
from pathlib import Path

from japanese_emotion_sentiment.experiment import main, run_experiment


def test_experiment_runner_is_reproducible() -> None:
    first = run_experiment(seed=42)
    second = run_experiment(seed=42)

    assert first == second
    assert first["schema_version"] == 1
    assert first["data"] == {
        "rows": 48,
        "train_rows": 28,
        "validation_rows": 10,
        "test_rows": 10,
    }


def test_experiment_cli_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    main(["--output", str(output), "--seed", "43"])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"] == {"seed": 43}
    assert len(payload["emotion_thresholds"]) == 8
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_experiment_compares_models_with_prior_baselines() -> None:
    result = run_experiment(seed=42)

    assert result["baseline_metrics"]["test"]["polarity_macro_f1"] < 1
    assert result["baseline_metrics"]["test"]["emotions"]["hamming_loss"] >= 0
    expected = (
        result["test_metrics"]["polarity_macro_f1"]
        - result["baseline_metrics"]["test"]["polarity_macro_f1"]
    )
    assert result["delta_vs_baseline"]["test_polarity_macro_f1"] == expected
