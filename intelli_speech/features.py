"""Spectral features shared by the GMM noise classifier and the ZF-Net mask estimator."""

from __future__ import annotations

import numpy as np

from .stft import STFTConfig, _import_librosa

DEFAULT_N_MELS = 40
LOG_EPSILON = 1e-6


def log_magnitude(stft: np.ndarray, *, epsilon: float = LOG_EPSILON) -> np.ndarray:
    """Return ``log(|X| + eps)`` as float32 with the STFT's (freq, time) shape."""
    spectrum = np.asarray(stft)
    if spectrum.ndim != 2:
        raise ValueError(f"Expected a two-dimensional STFT, received {spectrum.shape}.")
    if not np.isfinite(spectrum).all():
        raise ValueError("STFT contains non-finite values.")
    return np.log(np.abs(spectrum) + float(epsilon)).astype(np.float32)


def normalize_utterance(features: np.ndarray, *, epsilon: float = 1e-5) -> np.ndarray:
    """Zero-mean, unit-variance normalization over a whole utterance.

    Per-utterance normalization makes the network input independent of recording
    gain, which matters for microphone recordings from the frontend.
    """
    array = np.asarray(features, dtype=np.float32)
    mean = float(array.mean())
    std = float(array.std())
    return ((array - mean) / (std + float(epsilon))).astype(np.float32)


def log_mel_frames(
    stft: np.ndarray,
    config: STFTConfig = STFTConfig(),
    *,
    n_mels: int = DEFAULT_N_MELS,
    include_deltas: bool = True,
) -> np.ndarray:
    """Return per-frame log-mel features with shape (frames, n_mels[*2]).

    A single scalar (the utterance mean) is subtracted so the spectral shape of the
    background noise is preserved while overall gain is removed.
    """
    spectrum = np.asarray(stft)
    if spectrum.ndim != 2 or spectrum.shape[0] != config.freq_bins:
        raise ValueError(
            f"Expected an STFT with {config.freq_bins} frequency bins, "
            f"received {spectrum.shape}."
        )
    if int(n_mels) < 1:
        raise ValueError("n_mels must be positive.")

    librosa = _import_librosa()
    mel_basis = librosa.filters.mel(
        sr=config.sample_rate,
        n_fft=config.n_fft,
        n_mels=int(n_mels),
    )
    power = np.abs(spectrum) ** 2
    log_mel = np.log(mel_basis @ power + 1e-10).astype(np.float32)
    log_mel -= float(log_mel.mean())

    frames = [log_mel]
    if include_deltas:
        width = 9 if log_mel.shape[1] >= 9 else (log_mel.shape[1] // 2) * 2 + 1
        if width >= 3:
            frames.append(librosa.feature.delta(log_mel, width=width).astype(np.float32))
        else:
            frames.append(np.zeros_like(log_mel))
    return np.concatenate(frames, axis=0).T.astype(np.float32)
