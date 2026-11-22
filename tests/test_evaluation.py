from __future__ import annotations

import numpy as np
import pytest

from intelli_speech.evaluation import (
    calculate_snr_db,
    evaluate_enhancement,
    snr_improvement_db,
    summarize_metrics,
)


def test_calculate_snr_db_known_values() -> None:
    reference = np.ones(100, dtype=np.float64)
    estimate = reference + 0.1

    assert calculate_snr_db(reference, estimate) == pytest.approx(20.0)


def test_calculate_snr_db_identical_is_large() -> None:
    reference = np.ones(100, dtype=np.float64)

    assert calculate_snr_db(reference, reference) == pytest.approx(200.0)


def test_calculate_snr_db_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        calculate_snr_db(np.ones(10), np.ones(11))
    with pytest.raises(ValueError, match="zero power"):
        calculate_snr_db(np.zeros(10), np.ones(10))


def test_snr_improvement_db() -> None:
    clean = np.ones(100, dtype=np.float64)
    noisy = clean + 0.1
    enhanced = clean + 0.01

    assert snr_improvement_db(clean, noisy, enhanced) == pytest.approx(20.0)


def test_evaluate_enhancement_returns_expected_metrics() -> None:
    clean = np.ones(100, dtype=np.float32)
    noisy = clean + 0.1
    enhanced = clean - 0.01

    metrics = evaluate_enhancement(clean, noisy, enhanced, sample_rate=16000)

    assert metrics["input_snr_db"] == pytest.approx(20.0)
    assert metrics["output_snr_db"] == pytest.approx(40.0)
    assert metrics["snr_improvement_db"] == pytest.approx(20.0)
    assert metrics["length_samples"] == 100
    assert metrics["duration_sec"] == pytest.approx(100 / 16000)
    assert metrics["enhanced_clipping_detected"] == 0


def test_evaluate_enhancement_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        evaluate_enhancement(np.ones(10), np.ones(11), np.ones(10))


def test_summarize_metrics() -> None:
    rows = [
        {"score": 1.0, "other": 10.0},
        {"score": 3.0, "other": 20.0},
    ]

    summary = summarize_metrics(rows, metric_names=["score"])

    assert summary["count"] == 2
    score = summary["metrics"]["score"]
    assert score["mean"] == pytest.approx(2.0)
    assert score["median"] == pytest.approx(2.0)
    assert score["min"] == pytest.approx(1.0)
    assert score["max"] == pytest.approx(3.0)
    assert score["std"] == pytest.approx(1.0)


def test_summarize_metrics_ignores_none_values() -> None:
    rows = [
        {"score": 1.0, "optional": None},
        {"score": 3.0, "optional": None},
    ]

    summary = summarize_metrics(rows)

    assert set(summary["metrics"]) == {"score"}
    assert summary["metrics"]["score"]["count"] == 2


def test_summarize_metrics_rejects_non_numeric_and_empty() -> None:
    with pytest.raises(ValueError, match="not numeric"):
        summarize_metrics([{"score": "bad"}], metric_names=["score"])
    with pytest.raises(ValueError, match="empty"):
        summarize_metrics([])
