"""Tests for intelli_speech.gmm_noise."""
from __future__ import annotations

import numpy as np
import pytest

from intelli_speech.gmm_noise import NoiseGMM


def test_gmm_fit_two_classes_and_predict() -> None:
    """Fit GMM on two synthetic classes (N(0,1) vs N(3,1) in 4 dims) and predict correctly."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n_frames = 50
    dim = 4
    class_a = rng.standard_normal(size=(n_frames, dim)).astype(np.float64)
    class_b = rng.standard_normal(size=(n_frames, dim)) + 3.0  # N(3,1)

    gmm = NoiseGMM(random_state=0)
    gmm.fit({"noise_a": class_a, "noise_b": class_b})
    assert gmm.is_fitted

    # Predict on a sample from class A (should be "noise_a")
    test_a = rng.standard_normal(size=(5, dim)).astype(np.float64)
    pred = gmm.predict(test_a)
    assert pred == "noise_a"

    # Predict on a sample from class B (should be "noise_b")
    test_b = (rng.standard_normal(size=(5, dim)) + 3.0).astype(np.float64)
    pred = gmm.predict(test_b)
    assert pred == "noise_b"


def test_gmm_predict_proba_sums_to_one() -> None:
    """predict_proba returns probabilities that sum to 1."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n_frames = 30
    dim = 4
    class_a = rng.standard_normal(size=(n_frames, dim)).astype(np.float64)
    class_b = rng.standard_normal(size=(n_frames, dim)) + 3.0

    gmm = NoiseGMM(random_state=0)
    gmm.fit({"noise_a": class_a, "noise_b": class_b})

    proba = gmm.predict_proba(np.r_[class_a[:5], class_b[:5]])
    assert np.isclose(proba.sum(), 1.0, atol=1e-5)
    assert proba.dtype == np.float32


def test_gmm_calibrate_temperature_returns_grid_value() -> None:
    """calibrate_temperature returns a value from the default grid."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n_frames = 30
    dim = 4
    class_a = rng.standard_normal(size=(n_frames, dim)).astype(np.float64)
    class_b = rng.standard_normal(size=(n_frames, dim)) + 3.0

    gmm = NoiseGMM(random_state=0)
    gmm.fit({"noise_a": class_a, "noise_b": class_b})

    utterances = [np.r_[class_a[:10], class_b[:10]] for _ in range(3)]
    labels = ["noise_a", "noise_b", "noise_a"]
    temp = gmm.calibrate_temperature(utterances, labels)
    assert np.isclose(temp, gmm.temperature)


def test_gmm_calibrate_temperature_custom_grid() -> None:
    """calibrate_temperature accepts a custom grid of temperatures."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n_frames = 30
    dim = 4
    class_a = rng.standard_normal(size=(n_frames, dim)).astype(np.float64)
    class_b = rng.standard_normal(size=(n_frames, dim)) + 3.0

    gmm = NoiseGMM(random_state=0)
    gmm.fit({"noise_a": class_a, "noise_b": class_b})

    utterances = [np.r_[class_a[:10], class_b[:10]] for _ in range(3)]
    labels = ["noise_a", "noise_b", "noise_a"]
    temp = gmm.calibrate_temperature(utterances, labels, grid=[0.1, 1.0, 10.0])
    assert temp in [0.1, 1.0, 10.0]


def test_gmm_save_load_roundtrip_identical_proba(tmp_path) -> None:
    """save/load round-trip gives identical predict_proba."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n_frames = 30
    dim = 4
    class_a = rng.standard_normal(size=(n_frames, dim)).astype(np.float64)
    class_b = rng.standard_normal(size=(n_frames, dim)) + 3.0

    gmm = NoiseGMM(random_state=0)
    gmm.fit({"noise_a": class_a, "noise_b": class_b})

    proba_original = gmm.predict_proba(np.r_[class_a[:10], class_b[:10]])

    path = tmp_path / "gmm_checkpoint.joblib"
    gmm.save(path)
    loaded = NoiseGMM.load(path)

    proba_loaded = loaded.predict_proba(np.r_[class_a[:10], class_b[:10]])
    np.testing.assert_allclose(proba_original, proba_loaded, atol=1e-6)


def test_gmm_fit_one_class_raises_value_error() -> None:
    """fit with only one class raises ValueError."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    class_a = rng.standard_normal(size=(30, 4)).astype(np.float64)

    gmm = NoiseGMM(random_state=0)
    with pytest.raises(ValueError, match="At least two noise classes are required."):
        gmm.fit({"only_class": class_a})


def test_gmm_unfitted_save_raises_runtime_error(tmp_path) -> None:
    """save on unfitted GMM raises RuntimeError."""
    pytest.importorskip("sklearn")
    gmm = NoiseGMM(random_state=0)
    path = tmp_path / "empty_checkpoint.joblib"
    with pytest.raises(RuntimeError, match="Refusing to save an unfitted NoiseGMM"):
        gmm.save(path)