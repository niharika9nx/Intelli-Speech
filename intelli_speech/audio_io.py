from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class AudioLoadError(ValueError):
    """Raised when an audio file cannot be loaded as a valid mono recording."""


def resolve_metadata_audio_path(
    dataset_dir: str | Path,
    metadata_value: str | Path,
) -> Path:
    """Resolve an absolute or metadata-relative audio path without rewriting files."""
    dataset_path = Path(dataset_dir).expanduser()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    value = str(metadata_value).strip()
    if not value or value.lower() == "nan":
        raise ValueError("Audio metadata value is empty.")

    audio_path = Path(value).expanduser()
    if not audio_path.is_absolute():
        audio_path = dataset_path / audio_path

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return audio_path


def validate_mono_16khz(
    audio: np.ndarray,
    sample_rate: int,
    *,
    expected_sample_rate: int = 16000,
) -> np.ndarray:
    """Validate and return a one-dimensional finite mono waveform."""
    waveform = np.asarray(audio)
    if waveform.size == 0:
        raise AudioLoadError("Audio waveform is empty.")
    if not np.issubdtype(waveform.dtype, np.number):
        raise AudioLoadError("Audio waveform is not numeric.")
    if not np.isfinite(waveform).all():
        raise AudioLoadError("Audio waveform contains non-finite samples.")
    if waveform.ndim != 1:
        raise AudioLoadError(
            f"Expected mono audio with shape (samples,), received {waveform.shape}."
        )
    if int(sample_rate) != int(expected_sample_rate):
        raise AudioLoadError(
            f"Expected {expected_sample_rate} Hz, received {sample_rate} Hz."
        )
    return waveform.astype(np.float32, copy=False)


def load_audio(
    path: str | Path,
    *,
    expected_sample_rate: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load a WAV or other supported audio file and validate its format."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise AudioLoadError(
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
        raise AudioLoadError(f"Could not decode audio file {audio_path}: {exc}") from exc

    waveform = validate_mono_16khz(
        audio,
        sample_rate,
        expected_sample_rate=expected_sample_rate,
    )
    return waveform, int(sample_rate)


def write_audio(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    *,
    expected_sample_rate: int = 16000,
) -> Path:
    """Validate and write a mono 16 kHz waveform without modifying source files."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise AudioLoadError(
            "soundfile is required for audio writing."
        ) from exc

    waveform = validate_mono_16khz(
        audio,
        sample_rate,
        expected_sample_rate=expected_sample_rate,
    )
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), waveform, int(sample_rate), subtype="PCM_16")
    return output_path
