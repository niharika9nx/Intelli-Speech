from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stft import STFTConfig, compute_stft, inverse_stft


@dataclass(frozen=True)
class BaselineResult:
    """Enhanced waveform and the mask/gain used to produce it."""

    method: str
    enhanced: np.ndarray
    enhanced_stft: np.ndarray
    mask_or_gain: np.ndarray
    uses_clean_reference: bool
    noise_frames: int | None = None


def _validate_stft_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first_array = np.asarray(first)
    second_array = np.asarray(second)
    if first_array.shape != second_array.shape:
        raise ValueError(
            f"STFT shapes differ: {first_array.shape} vs {second_array.shape}"
        )
    if first_array.ndim != 2:
        raise ValueError("Expected two-dimensional STFT arrays.")
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        raise ValueError("STFT arrays contain non-finite values.")
    return first_array, second_array


def oracle_ratio_mask(
    clean_stft: np.ndarray,
    noise_stft: np.ndarray,
    *,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute the oracle ideal ratio mask from clean and noise STFTs."""
    if float(epsilon) <= 0:
        raise ValueError("epsilon must be positive.")
    clean_array, noise_array = _validate_stft_pair(clean_stft, noise_stft)
    clean_magnitude = np.abs(clean_array)
    noise_magnitude = np.abs(noise_array)
    mask = clean_magnitude / (
        clean_magnitude + noise_magnitude + float(epsilon)
    )
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def apply_mask(noisy_stft: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a real-valued mask while preserving the noisy phase."""
    noisy_array, mask_array = _validate_stft_pair(noisy_stft, mask)
    if not np.isrealobj(mask_array):
        raise ValueError("The mask must be real-valued.")
    if np.any(mask_array < 0) or np.any(mask_array > 1):
        raise ValueError("The mask values must lie in [0, 1].")
    return (noisy_array * mask_array).astype(np.complex64)


def estimate_noise_power(
    noisy_stft: np.ndarray,
    noise_frames: int,
) -> np.ndarray:
    """Estimate per-frequency noise power from the first N STFT frames."""
    spectrum = np.asarray(noisy_stft)
    if spectrum.ndim != 2:
        raise ValueError("Expected a two-dimensional noisy STFT.")
    if not np.isfinite(spectrum).all():
        raise ValueError("Noisy STFT contains non-finite values.")
    if int(noise_frames) < 1:
        raise ValueError("noise_frames must be a positive integer.")

    frames = min(int(noise_frames), spectrum.shape[1])
    power = np.mean(
        np.abs(spectrum[:, :frames]) ** 2,
        axis=1,
        keepdims=True,
    )
    return power.astype(np.float32)


def wiener_gain(
    noisy_stft: np.ndarray,
    noise_power: np.ndarray,
    *,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute a non-oracle Wiener-style gain from noisy power and noise power."""
    if float(epsilon) <= 0:
        raise ValueError("epsilon must be positive.")
    spectrum = np.asarray(noisy_stft)
    if spectrum.ndim != 2:
        raise ValueError("Expected a two-dimensional noisy STFT.")
    if not np.isfinite(spectrum).all():
        raise ValueError("Noisy STFT contains non-finite values.")

    power = np.asarray(noise_power, dtype=np.float64)
    if power.ndim == 1:
        power = power[:, None]
    if power.ndim != 2:
        raise ValueError("noise_power must be one- or two-dimensional.")
    if power.shape[0] != spectrum.shape[0]:
        raise ValueError(
            f"noise_power has {power.shape[0]} bins; expected {spectrum.shape[0]}."
        )
    if power.shape[1] not in (1, spectrum.shape[1]):
        raise ValueError(
            "noise_power must have one column or match the STFT frame count."
        )
    if not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("noise_power must be finite and non-negative.")

    noisy_power = np.abs(spectrum) ** 2
    gain = np.maximum(
        0.0,
        1.0 - power / (noisy_power + float(epsilon)),
    )
    return np.clip(gain, 0.0, 1.0).astype(np.float32)


def wiener_enhance(
    noisy: np.ndarray,
    config: STFTConfig = STFTConfig(),
    *,
    noise_duration_sec: float = 0.25,
) -> BaselineResult:
    """Run the non-oracle Wiener-style baseline on a noisy waveform."""
    waveform = np.asarray(noisy, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Expected a non-empty one-dimensional noisy waveform.")
    if not np.isfinite(waveform).all():
        raise ValueError("Noisy waveform contains non-finite samples.")
    if float(noise_duration_sec) <= 0:
        raise ValueError("noise_duration_sec must be positive.")

    noisy_stft = compute_stft(waveform, config)
    noise_samples = min(
        len(waveform),
        int(round(float(noise_duration_sec) * config.sample_rate)),
    )
    if noise_samples < 1:
        raise ValueError("Noise estimation segment is empty.")
    noise_frames = max(1, int(np.ceil(noise_samples / config.hop_length)))
    noise_frames = min(noise_frames, noisy_stft.shape[1])
    noise_power = estimate_noise_power(noisy_stft, noise_frames)
    gain = wiener_gain(noisy_stft, noise_power)
    enhanced_stft = (gain * noisy_stft).astype(np.complex64)
    enhanced = inverse_stft(enhanced_stft, config, length=len(waveform))

    return BaselineResult(
        method="wiener",
        enhanced=enhanced,
        enhanced_stft=enhanced_stft,
        mask_or_gain=gain,
        uses_clean_reference=False,
        noise_frames=noise_frames,
    )


def oracle_irm_enhance(
    clean: np.ndarray,
    noisy: np.ndarray,
    config: STFTConfig = STFTConfig(),
) -> BaselineResult:
    """Run the oracle IRM baseline; this uses the clean reference by design."""
    clean_array = np.asarray(clean, dtype=np.float32)
    noisy_array = np.asarray(noisy, dtype=np.float32)
    if clean_array.shape != noisy_array.shape:
        raise ValueError(
            f"Clean/noisy shapes differ: {clean_array.shape} vs {noisy_array.shape}"
        )
    if clean_array.ndim != 1 or clean_array.size == 0:
        raise ValueError("Expected non-empty one-dimensional waveforms.")
    if not np.isfinite(clean_array).all() or not np.isfinite(noisy_array).all():
        raise ValueError("Clean or noisy waveform contains non-finite samples.")

    clean_stft = compute_stft(clean_array, config)
    noisy_stft = compute_stft(noisy_array, config)
    noise_stft = noisy_stft - clean_stft
    mask = oracle_ratio_mask(clean_stft, noise_stft)
    enhanced_stft = apply_mask(noisy_stft, mask)
    enhanced = inverse_stft(enhanced_stft, config, length=len(noisy_array))

    return BaselineResult(
        method="oracle-irm",
        enhanced=enhanced,
        enhanced_stft=enhanced_stft,
        mask_or_gain=mask,
        uses_clean_reference=True,
    )


__all__ = [
    "BaselineResult",
    "apply_mask",
    "estimate_noise_power",
    "oracle_irm_enhance",
    "oracle_ratio_mask",
    "wiener_enhance",
    "wiener_gain",
]
