from __future__ import annotations

import numpy as np


def peak_amplitude(audio: np.ndarray) -> float:
    """Return the maximum absolute sample amplitude."""
    waveform = np.asarray(audio, dtype=np.float64)
    if waveform.size == 0:
        raise ValueError("Audio waveform is empty.")
    if not np.isfinite(waveform).all():
        raise ValueError("Audio waveform contains non-finite samples.")
    return float(np.max(np.abs(waveform)))


def rms_amplitude(audio: np.ndarray) -> float:
    """Return the root-mean-square amplitude."""
    waveform = np.asarray(audio, dtype=np.float64)
    if waveform.size == 0:
        raise ValueError("Audio waveform is empty.")
    if not np.isfinite(waveform).all():
        raise ValueError("Audio waveform contains non-finite samples.")
    return float(np.sqrt(np.mean(np.square(waveform))))


def compute_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute signal-to-noise ratio from aligned waveform arrays."""
    signal_array = np.asarray(signal, dtype=np.float64)
    noise_array = np.asarray(noise, dtype=np.float64)

    if signal_array.shape != noise_array.shape:
        raise ValueError(
            f"Signal/noise shapes differ: {signal_array.shape} vs {noise_array.shape}"
        )
    if signal_array.size == 0:
        raise ValueError("Signal and noise waveforms are empty.")
    if not np.isfinite(signal_array).all() or not np.isfinite(noise_array).all():
        raise ValueError("Signal or noise contains non-finite samples.")

    signal_power = np.mean(np.square(signal_array))
    noise_power = np.mean(np.square(noise_array))
    if signal_power <= 0:
        raise ValueError("Signal power must be greater than zero.")
    if noise_power <= 0:
        raise ValueError("Noise power must be greater than zero.")

    return float(10.0 * np.log10(signal_power / noise_power))


def clipping_report(
    audio: np.ndarray,
    *,
    full_scale_threshold: float = 1.0,
) -> dict[str, float | int | bool]:
    """Summarize peak level and full-scale clipping candidates."""
    waveform = np.asarray(audio)
    peak = peak_amplitude(waveform)
    full_scale_samples = int(np.count_nonzero(np.abs(waveform) >= full_scale_threshold))
    return {
        "peak_amplitude": peak,
        "full_scale_sample_count": full_scale_samples,
        "clipping_detected": full_scale_samples > 0,
    }
