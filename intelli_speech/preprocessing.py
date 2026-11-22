from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np

from .audio_io import write_audio
from .metrics import compute_snr_db


class PreprocessingDependencyError(RuntimeError):
    """Raised when an optional preprocessing dependency is unavailable."""


@dataclass(frozen=True)
class MixResult:
    """Clean target and noisy mixture produced at a requested SNR."""

    clean: np.ndarray
    mixture: np.ndarray
    scaled_noise: np.ndarray
    measured_snr_db: float
    common_gain: float
    noise_gain: float


def to_mono_16k(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_sample_rate: int = 16000,
) -> np.ndarray:
    """Convert a waveform to mono float32 at the target sample rate."""
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError("Audio waveform is empty.")
    if not np.isfinite(waveform).all():
        raise ValueError("Audio waveform contains non-finite samples.")
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive.")
    if int(target_sample_rate) <= 0:
        raise ValueError("target_sample_rate must be positive.")

    if waveform.ndim == 2:
        # soundfile returns (samples, channels); also accept (channels, samples).
        waveform = (
            waveform.mean(axis=1)
            if waveform.shape[1] <= waveform.shape[0]
            else waveform.mean(axis=0)
        )
    elif waveform.ndim != 1:
        raise ValueError(
            f"Expected a mono or two-dimensional waveform, received {waveform.shape}."
        )

    waveform = waveform.astype(np.float32, copy=False)
    if int(sample_rate) != int(target_sample_rate):
        divisor = gcd(int(sample_rate), int(target_sample_rate))
        waveform = resample_poly(
            waveform,
            int(target_sample_rate) // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)

    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Resampled audio is empty or contains non-finite samples.")
    return waveform


def resample_poly(audio: np.ndarray, up: int, down: int) -> np.ndarray:
    """Resample with SciPy while keeping the dependency local to this function."""
    try:
        from scipy.signal import resample_poly as scipy_resample_poly
    except ImportError as exc:
        raise PreprocessingDependencyError(
            "scipy is required for resampling."
        ) from exc
    return scipy_resample_poly(audio, up, down)


def load_wav_resampled(
    path: str | Path,
    *,
    target_sample_rate: int = 16000,
) -> np.ndarray:
    """Read any supported WAV and convert it to mono at the target rate."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise PreprocessingDependencyError(
            "soundfile is required for audio loading."
        ) from exc

    audio_path = Path(path).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    try:
        audio, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False,
        )
    except Exception as exc:
        raise ValueError(f"Could not decode audio file {audio_path}: {exc}") from exc
    return to_mono_16k(
        audio,
        sample_rate,
        target_sample_rate=target_sample_rate,
    )


def trim_silence(
    audio: np.ndarray,
    *,
    sample_rate: int = 16000,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    pad_seconds: float = 0.150,
    min_duration_sec: float = 0.5,
) -> np.ndarray:
    """Trim only leading/trailing silence while preserving internal pauses."""
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Expected a non-empty one-dimensional waveform.")
    if not np.isfinite(waveform).all():
        raise ValueError("Audio waveform contains non-finite samples.")
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive.")
    if pad_seconds < 0:
        raise ValueError("pad_seconds must not be negative.")

    try:
        import librosa
    except Exception as exc:
        raise PreprocessingDependencyError(
            "librosa is required for silence trimming. Install a NumPy-compatible "
            "librosa/numba pair before using this function."
        ) from exc

    intervals = librosa.effects.split(
        waveform,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if len(intervals) == 0:
        raise ValueError("No non-silent speech detected.")

    padding = int(round(pad_seconds * int(sample_rate)))
    start = max(0, int(intervals[0, 0]) - padding)
    end = min(len(waveform), int(intervals[-1, 1]) + padding)
    trimmed = waveform[start:end].astype(np.float32, copy=False)

    minimum = int(round(min_duration_sec * int(sample_rate)))
    if len(trimmed) < minimum:
        raise ValueError(
            f"Speech too short after trimming: {len(trimmed) / sample_rate:.3f}s"
        )
    return trimmed


def get_noise_segment(
    noise: np.ndarray,
    length: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select a DC-removed noise segment of exactly ``length`` samples."""
    waveform = np.asarray(noise, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Noise must be a non-empty one-dimensional waveform.")
    if not np.isfinite(waveform).all():
        raise ValueError("Noise contains non-finite samples.")
    if int(length) < 1:
        raise ValueError("length must be a positive integer.")

    target_length = int(length)
    if len(waveform) < target_length:
        waveform = np.tile(
            waveform,
            int(np.ceil(target_length / len(waveform))),
        )
    if rng is None:
        rng = np.random.default_rng()
    start = int(rng.integers(0, len(waveform) - target_length + 1))
    segment = waveform[start : start + target_length].astype(np.float32, copy=True)
    segment -= np.mean(segment)

    rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
    if not np.isfinite(rms) or rms < 1e-7:
        raise ValueError("Selected noise segment is effectively silent.")
    return segment


def _scaled_noise(
    clean: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
) -> tuple[np.ndarray, float]:
    clean_array = np.asarray(clean, dtype=np.float64)
    noise_array = np.asarray(noise, dtype=np.float64)
    if clean_array.shape != noise_array.shape:
        raise ValueError(
            f"Clean/noise shapes differ: {clean_array.shape} vs {noise_array.shape}"
        )
    if clean_array.size == 0:
        raise ValueError("Clean and noise waveforms are empty.")
    if not np.isfinite(clean_array).all() or not np.isfinite(noise_array).all():
        raise ValueError("Clean or noise contains non-finite samples.")
    if not np.isfinite(float(target_snr_db)):
        raise ValueError("target_snr_db must be finite.")

    clean_rms = float(np.sqrt(np.mean(clean_array**2)))
    noise_rms = float(np.sqrt(np.mean(noise_array**2)))
    if clean_rms < 1e-7 or noise_rms < 1e-7:
        raise ValueError("Clean or noise segment has negligible RMS.")

    noise_gain = clean_rms / (noise_rms * (10 ** (float(target_snr_db) / 20.0)))
    scaled = noise_array * noise_gain
    return scaled.astype(np.float32), float(noise_gain)


def scale_noise_to_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
) -> np.ndarray:
    """Scale noise to a target SNR and return the unclipped mixture."""
    scaled_noise, _ = _scaled_noise(clean, noise, target_snr_db)
    mixture = np.asarray(clean, dtype=np.float32) + scaled_noise
    return mixture.astype(np.float32)


def measured_snr_db(clean: np.ndarray, mixture: np.ndarray) -> float:
    """Measure SNR from an aligned clean target and noisy mixture."""
    clean_array = np.asarray(clean, dtype=np.float64)
    mixture_array = np.asarray(mixture, dtype=np.float64)
    if clean_array.shape != mixture_array.shape:
        raise ValueError(
            f"Clean/mixture shapes differ: {clean_array.shape} vs {mixture_array.shape}"
        )
    residual = mixture_array - clean_array
    return compute_snr_db(clean_array, residual)


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
    *,
    peak_limit: float = 0.99,
) -> MixResult:
    """Mix clean speech and noise at a target SNR with a shared peak gain."""
    if not 0 < float(peak_limit) <= 1:
        raise ValueError("peak_limit must be in (0, 1].")

    clean_array = np.asarray(clean, dtype=np.float32)
    scaled_noise, noise_gain = _scaled_noise(clean_array, noise, target_snr_db)
    mixture = (clean_array.astype(np.float64) + scaled_noise.astype(np.float64))

    peak = float(np.max(np.abs(mixture)))
    common_gain = min(1.0, float(peak_limit) / peak) if peak > 0 else 1.0
    clean_out = (clean_array * common_gain).astype(np.float32)
    mixture_out = (mixture * common_gain).astype(np.float32)
    measured = measured_snr_db(clean_array, mixture.astype(np.float32))

    return MixResult(
        clean=clean_out,
        mixture=mixture_out,
        scaled_noise=(scaled_noise * common_gain).astype(np.float32),
        measured_snr_db=float(measured),
        common_gain=float(common_gain),
        noise_gain=float(noise_gain),
    )


def write_pcm16(
    path: str | Path,
    audio: np.ndarray,
    *,
    sample_rate: int = 16000,
) -> Path:
    """Validate and write a non-clipping mono PCM16 WAV."""
    waveform = np.asarray(audio)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Audio waveform is empty or contains non-finite samples.")
    if float(np.max(np.abs(waveform))) >= 1.0:
        raise ValueError(f"Potential clipping; refusing to write: {path}")
    return write_audio(path, waveform, sample_rate)


__all__ = [
    "MixResult",
    "PreprocessingDependencyError",
    "get_noise_segment",
    "load_wav_resampled",
    "measured_snr_db",
    "mix_at_snr",
    "scale_noise_to_snr",
    "to_mono_16k",
    "trim_silence",
    "write_pcm16",
]
