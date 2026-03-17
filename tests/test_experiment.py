from __future__ import annotations

import json
from pathlib import Path

from japanese_emotion_sentiment.data import normalize_text
from japanese_emotion_sentiment.experiment import (
    main,
    punctuation_variant,
    run_experiment,
    unicode_variant,
)


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
    assert payload["config"] == {
        "feature_mode": "char",
        "fixed_threshold": 0.5,
        "punctuation_style": "mixed",
        "seed": 43,
        "stress_fraction": 0.0,
        "stress_mode": "none",
        "threshold_mode": "tuned",
    }
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


def test_char_word_ablation_builds_distinct_feature_spaces() -> None:
    results = {
        mode: run_experiment(seed=42, feature_mode=mode) for mode in ("char", "word", "char_word")
    }

    assert all(result["feature_count"] > 0 for result in results.values())
    assert results["char_word"]["feature_count"] == (
        results["char"]["feature_count"] + results["word"]["feature_count"]
    )
    assert all(0 <= result["test_metrics"]["polarity_macro_f1"] <= 1 for result in results.values())


def test_unicode_stress_is_recovered_by_nfkc_normalization() -> None:
    text = "テストABC123"
    stressed = unicode_variant(text)
    result = run_experiment(seed=42, stress_mode="unicode", stress_fraction=1.0)

    assert stressed != text
    assert normalize_text(stressed) == text
    assert result["stress_diagnostics"]["selected_rows"] == 10
    assert result["stress_diagnostics"]["normalized_recovered_rows"] == 10
    assert result["stress_diagnostics"]["normalized_metrics"] == result["test_metrics"]


def test_punctuation_stress_changes_selected_holdout_texts() -> None:
    assert punctuation_variant("驚いた。", "remove") == "驚いた"
    assert punctuation_variant("驚いた。", "emoji").endswith("🙂")

    result = run_experiment(
        seed=42,
        stress_mode="punctuation",
        stress_fraction=0.5,
        punctuation_style="remove",
    )

    assert result["stress_diagnostics"]["selected_rows"] == 5
    assert result["stress_diagnostics"]["style"] == "remove"
    assert result["stress_diagnostics"]["stressed_metrics"] == result["test_metrics"]


def test_per_emotion_diagnostics_cover_all_labels_and_thresholds() -> None:
    result = run_experiment(seed=42)
    diagnostics = result["per_emotion"]["test"]
    thresholds = result["emotion_thresholds"]

    assert len(diagnostics) == 8
    assert set(diagnostics) == set(thresholds)
    assert all(0 <= row["precision"] <= 1 for row in diagnostics.values())
    assert all(row["threshold"] == thresholds[name] for name, row in diagnostics.items())


def test_fixed_threshold_mode_preserves_tuning_diagnostics() -> None:
    result = run_experiment(seed=42, threshold_mode="fixed", fixed_threshold=0.7)

    assert set(result["emotion_thresholds"].values()) == {0.7}
    assert result["emotion_thresholds"] != result["tuned_emotion_thresholds"]
    assert set(result["threshold_diagnostics"]) == set(result["emotion_thresholds"])
    assert all(
        len(diagnostics["curve"]) >= 19
        and diagnostics["tuned_threshold"] == result["tuned_emotion_thresholds"][emotion]
        for emotion, diagnostics in result["threshold_diagnostics"].items()
    )
