from __future__ import annotations

import numpy as np
import pytest

from intelli_speech.baselines import (
    apply_mask,
    estimate_noise_power,
    oracle_irm_enhance,
    oracle_ratio_mask,
    wiener_enhance,
    wiener_gain,
)
from intelli_speech.stft import STFTConfig


def test_oracle_ratio_mask_known_values() -> None:
    clean = np.array([[2.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64)
    noise = np.array([[1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64)

    mask = oracle_ratio_mask(clean, noise)

    np.testing.assert_allclose(mask, [[2.0 / 3.0, 0.0]], atol=1e-6)


def test_oracle_ratio_mask_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        oracle_ratio_mask(np.ones((2, 3), dtype=np.complex64), np.ones((2, 4)))


def test_apply_mask_preserves_phase_and_zeroes_silence() -> None:
    noisy = np.array([[1.0 + 1.0j, 2.0 + 0.0j]], dtype=np.complex64)
    mask = np.array([[0.5, 0.0]], dtype=np.float32)

    result = apply_mask(noisy, mask)

    np.testing.assert_allclose(result, [[0.5 + 0.5j, 0.0]], atol=1e-6)


def test_apply_mask_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_mask(np.ones((2, 2), dtype=np.complex64), np.full((2, 2), 1.1))


def test_estimate_noise_power_uses_leading_frames() -> None:
    spectrum = np.ones((3, 4), dtype=np.complex64)
    spectrum[:, :2] = 2.0

    power = estimate_noise_power(spectrum, 2)

    np.testing.assert_allclose(power, np.full((3, 1), 4.0), atol=1e-6)


def test_wiener_gain_known_values_and_clipping() -> None:
    noisy = np.ones((3, 4), dtype=np.complex64)
    noise_power = np.array([[0.25], [0.5], [2.0]])

    gain = wiener_gain(noisy, noise_power)

    np.testing.assert_allclose(
        gain[:, 0],
        [0.75, 0.5, 0.0],
        atol=1e-6,
    )
    assert np.all(gain >= 0)
    assert np.all(gain <= 1)


def test_wiener_gain_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="expected 3"):
        wiener_gain(
            np.ones((3, 4), dtype=np.complex64),
            np.ones((2, 1)),
        )


def test_wiener_enhance_returns_finite_output() -> None:
    pytest.importorskip("librosa")
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    clean = (0.3 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    rng = np.random.default_rng(5)
    noisy = (clean + 0.03 * rng.normal(size=clean.shape)).astype(np.float32)
    config = STFTConfig(n_fft=256, hop_length=64)

    result = wiener_enhance(noisy, config, noise_duration_sec=0.25)

    assert result.method == "wiener"
    assert result.uses_clean_reference is False
    assert result.enhanced.shape == noisy.shape
    assert np.isfinite(result.enhanced).all()
    assert result.mask_or_gain.min() >= 0
    assert result.mask_or_gain.max() <= 1
    assert result.noise_frames is not None and result.noise_frames >= 1


def test_oracle_irm_enhance_returns_finite_output() -> None:
    pytest.importorskip("librosa")
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    clean = (0.3 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    rng = np.random.default_rng(9)
    noisy = (clean + 0.03 * rng.normal(size=clean.shape)).astype(np.float32)
    config = STFTConfig(n_fft=256, hop_length=64)

    result = oracle_irm_enhance(clean, noisy, config)

    assert result.method == "oracle-irm"
    assert result.uses_clean_reference is True
    assert result.enhanced.shape == noisy.shape
    assert np.isfinite(result.enhanced).all()
    assert result.mask_or_gain.min() >= 0
    assert result.mask_or_gain.max() <= 1
