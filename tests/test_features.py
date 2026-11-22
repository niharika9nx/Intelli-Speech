"""Tests for intelli_speech.features."""
from __future__ import annotations

import numpy as np
import pytest

from intelli_speech.features import log_magnitude, log_mel_frames, normalize_utterance
from intelli_speech.stft import STFTConfig


def _noise_stft(seconds: float = 1.0):
    from intelli_speech.stft import compute_stft

    waveform = np.random.default_rng(42).normal(size=int(16000 * seconds)).astype(np.float32)
    return compute_stft(waveform * 0.1, STFTConfig())


def test_log_magnitude_shape_and_dtype() -> None:
    spectrum = (np.ones((257, 12)) + 1j * np.ones((257, 12))).astype(np.complex64)
    result = log_magnitude(spectrum)
    assert result.shape == (257, 12)
    assert result.dtype == np.float32
    assert np.allclose(result, np.log(np.sqrt(2.0) + 1e-6), atol=1e-5)


def test_log_magnitude_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        log_magnitude(np.ones(10))


def test_normalize_utterance_zero_mean_unit_std() -> None:
    features = np.random.default_rng(0).normal(3.0, 2.0, size=(257, 40))
    result = normalize_utterance(features)
    assert result.dtype == np.float32
    assert abs(float(result.mean())) < 1e-4
    assert abs(float(result.std()) - 1.0) < 1e-3


def test_normalize_utterance_constant_input() -> None:
    result = normalize_utterance(np.full(10, 5.0))
    # Zero variance: the epsilon guard maps a constant utterance to all zeros.
    assert np.allclose(result, 0.0, atol=1e-5)
    assert np.isfinite(result).all()


def test_log_mel_frames_with_deltas_has_80_columns() -> None:
    pytest.importorskip("librosa")
    stft = _noise_stft()
    frames = log_mel_frames(stft)
    assert frames.shape == (stft.shape[1], 80)
    assert frames.dtype == np.float32
    assert np.isfinite(frames).all()
    assert abs(float(frames[:, :40].mean())) < 1e-4  # scalar mean removed


def test_log_mel_frames_without_deltas_has_40_columns() -> None:
    pytest.importorskip("librosa")
    stft = _noise_stft()
    assert log_mel_frames(stft, include_deltas=False).shape == (stft.shape[1], 40)


def test_log_mel_frames_very_short_utterance() -> None:
    pytest.importorskip("librosa")
    stft = _noise_stft(seconds=0.02)  # only a few frames
    frames = log_mel_frames(stft)
    assert frames.shape[1] == 80
    assert np.isfinite(frames).all()


def test_log_mel_frames_rejects_wrong_freq_bins() -> None:
    with pytest.raises(ValueError, match="257 frequency bins"):
        log_mel_frames(np.ones((200, 10)))
