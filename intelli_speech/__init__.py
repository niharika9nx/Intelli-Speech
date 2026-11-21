"""Reusable utilities for the Intelli-Speech research pipeline."""

from .audio_io import (
    AudioLoadError,
    load_audio,
    resolve_metadata_audio_path,
    validate_mono_16khz,
    write_audio,
)
from .metrics import (
    clipping_report,
    compute_snr_db,
    peak_amplitude,
    rms_amplitude,
)

__all__ = [
    "AudioLoadError",
    "load_audio",
    "resolve_metadata_audio_path",
    "validate_mono_16khz",
    "write_audio",
    "clipping_report",
    "compute_snr_db",
    "peak_amplitude",
    "rms_amplitude",
]
