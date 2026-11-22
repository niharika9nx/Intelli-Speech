from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class STFTDependencyError(RuntimeError):
    """Raised when librosa cannot be imported in the current runtime."""


@dataclass(frozen=True)
class STFTConfig:
    """Short-time Fourier transform settings used by the research pipeline."""

    n_fft: int = 512
    hop_length: int = 128
    sample_rate: int = 16000
    window: str = "hann"
    center: bool = True

    def __post_init__(self) -> None:
        if int(self.n_fft) < 2:
            raise ValueError("n_fft must be at least 2.")
        if int(self.hop_length) < 1:
            raise ValueError("hop_length must be positive.")
        if int(self.hop_length) > int(self.n_fft):
            raise ValueError("hop_length must not exceed n_fft.")
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive.")
        if not str(self.window).strip():
            raise ValueError("window must be a non-empty string.")

    @property
    def freq_bins(self) -> int:
        """Number of frequency bins produced by the real-valued STFT."""
        return int(self.n_fft) // 2 + 1


@dataclass(frozen=True)
class STFTCheckpoint:
    """Arrays and settings persisted for reproducible STFT experiments."""

    clean: np.ndarray
    noisy: np.ndarray
    clean_stft: np.ndarray
    noisy_stft: np.ndarray
    sample_rate: int
    n_fft: int
    hop_length: int

    @property
    def config(self) -> STFTConfig:
        return STFTConfig(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
        )


def _import_librosa():
    try:
        import librosa
    except Exception as exc:
        raise STFTDependencyError(
            "librosa is required for STFT operations. In Colab, install a "
            "NumPy-compatible numba/librosa pair (NumPy <=2.4) before running."
        ) from exc
    return librosa


def compute_stft(
    waveform: np.ndarray,
    config: STFTConfig = STFTConfig(),
) -> np.ndarray:
    """Compute a complex STFT with the configured librosa settings."""
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Expected a non-empty one-dimensional waveform.")
    if not np.isfinite(audio).all():
        raise ValueError("Waveform contains non-finite samples.")

    librosa = _import_librosa()
    return np.asarray(
        librosa.stft(
            audio,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            window=config.window,
            center=config.center,
        )
    )


def inverse_stft(
    stft: np.ndarray,
    config: STFTConfig = STFTConfig(),
    *,
    length: int | None = None,
) -> np.ndarray:
    """Reconstruct a waveform from a complex STFT."""
    spectrum = np.asarray(stft)
    if spectrum.ndim != 2:
        raise ValueError(f"Expected a two-dimensional STFT, received {spectrum.shape}.")
    if spectrum.shape[0] != config.freq_bins:
        raise ValueError(
            f"Expected {config.freq_bins} frequency bins, received {spectrum.shape[0]}."
        )
    if not np.isfinite(spectrum).all():
        raise ValueError("STFT contains non-finite values.")
    if length is not None and int(length) < 1:
        raise ValueError("length must be a positive integer when provided.")

    librosa = _import_librosa()
    waveform = librosa.istft(
        spectrum,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window=config.window,
        center=config.center,
        length=length,
    )
    result = np.asarray(waveform, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Inverse STFT produced non-finite samples.")
    return result


def reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> dict[str, float]:
    """Compare a waveform with its STFT/iSTFT reconstruction."""
    reference = np.asarray(original, dtype=np.float64)
    estimate = np.asarray(reconstructed, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError(
            f"Waveform shapes differ: {reference.shape} vs {estimate.shape}"
        )
    if reference.size == 0:
        raise ValueError("Waveforms are empty.")
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
        raise ValueError("Waveforms contain non-finite samples.")

    error = reference - estimate
    error_power = float(np.mean(error**2))
    signal_power = float(np.mean(reference**2))
    return {
        "rmse": float(np.sqrt(error_power)),
        "max_abs_error": float(np.max(np.abs(error))),
        "error_to_signal_snr_db": float(
            10.0 * np.log10(signal_power / max(error_power, 1e-20))
        )
        if signal_power > 0
        else float("inf"),
    }


def stft_round_trip(
    waveform: np.ndarray,
    config: STFTConfig = STFTConfig(),
) -> tuple[np.ndarray, dict[str, float]]:
    """Run STFT then iSTFT and return the reconstruction plus error metrics."""
    spectrum = compute_stft(waveform, config)
    reconstructed = inverse_stft(spectrum, config, length=len(waveform))
    return reconstructed, reconstruction_error(waveform, reconstructed)


def load_stft_checkpoint(path: str | Path) -> STFTCheckpoint:
    """Load and validate the persisted STFT analysis checkpoint."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"STFT checkpoint not found: {checkpoint_path}")

    required = {
        "clean",
        "noisy",
        "clean_stft",
        "noisy_stft",
        "sample_rate",
        "n_fft",
        "hop_length",
    }
    try:
        with np.load(checkpoint_path, allow_pickle=False) as data:
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    f"STFT checkpoint is missing keys: {sorted(missing)}"
                )
            clean = data["clean"].copy()
            noisy = data["noisy"].copy()
            clean_stft = data["clean_stft"].copy()
            noisy_stft = data["noisy_stft"].copy()
            sample_rate = int(data["sample_rate"])
            n_fft = int(data["n_fft"])
            hop_length = int(data["hop_length"])
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read STFT checkpoint {checkpoint_path}: {exc}") from exc

    if clean.ndim != 1 or noisy.ndim != 1:
        raise ValueError("Checkpoint clean/noisy waveforms must be one-dimensional.")
    if clean.shape != noisy.shape:
        raise ValueError(
            f"Checkpoint clean/noisy lengths differ: {clean.shape} vs {noisy.shape}"
        )
    if clean.size == 0 or not np.isfinite(clean).all() or not np.isfinite(noisy).all():
        raise ValueError("Checkpoint waveforms are empty or contain non-finite values.")
    if clean_stft.ndim != 2 or noisy_stft.ndim != 2:
        raise ValueError("Checkpoint STFT arrays must be two-dimensional.")
    if clean_stft.shape != noisy_stft.shape:
        raise ValueError(
            f"Checkpoint STFT shapes differ: {clean_stft.shape} vs {noisy_stft.shape}"
        )
    if not np.isfinite(clean_stft).all() or not np.isfinite(noisy_stft).all():
        raise ValueError("Checkpoint STFT arrays contain non-finite values.")
    if sample_rate <= 0 or n_fft < 2 or hop_length < 1:
        raise ValueError("Checkpoint STFT settings are invalid.")
    if clean_stft.shape[0] != n_fft // 2 + 1:
        raise ValueError(
            f"Checkpoint has {clean_stft.shape[0]} frequency bins for n_fft={n_fft}."
        )

    return STFTCheckpoint(
        clean=clean,
        noisy=noisy,
        clean_stft=clean_stft,
        noisy_stft=noisy_stft,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )


def save_stft_checkpoint(
    path: str | Path,
    checkpoint: STFTCheckpoint,
) -> Path:
    """Persist an STFT checkpoint without writing into a source dataset."""
    output_path = Path(path).expanduser()
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_name(output_path.name + ".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        clean=np.asarray(checkpoint.clean),
        noisy=np.asarray(checkpoint.noisy),
        clean_stft=np.asarray(checkpoint.clean_stft),
        noisy_stft=np.asarray(checkpoint.noisy_stft),
        sample_rate=np.asarray(checkpoint.sample_rate),
        n_fft=np.asarray(checkpoint.n_fft),
        hop_length=np.asarray(checkpoint.hop_length),
    )
    return output_path


__all__ = [
    "STFTConfig",
    "STFTCheckpoint",
    "STFTDependencyError",
    "compute_stft",
    "inverse_stft",
    "load_stft_checkpoint",
    "reconstruction_error",
    "save_stft_checkpoint",
    "stft_round_trip",
]
