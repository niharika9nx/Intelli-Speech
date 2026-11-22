from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class ClearVoiceDependencyError(RuntimeError):
    """Raised when ClearerVoice cannot be imported in the current environment."""


@dataclass(frozen=True)
class FRCRNConfig:
    """ClearerVoice FRCRN configuration used by the research pipeline."""

    task: str = "speech_enhancement"
    model_names: tuple[str, ...] = ("FRCRN_SE_16K",)
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        if not str(self.task).strip():
            raise ValueError("task must be a non-empty string.")
        if not self.model_names or any(
            not str(name).strip() for name in self.model_names
        ):
            raise ValueError("model_names must contain at least one non-empty name.")
        if int(self.sample_rate) != 16000:
            raise ValueError("FRCRN_SE_16K requires 16 kHz audio.")


@dataclass(frozen=True)
class FRCRNResult:
    """Enhanced waveform and provenance for one FRCRN inference call."""

    enhanced: np.ndarray
    elapsed_sec: float
    input_path: Path
    model_names: tuple[str, ...]
    output_peak: float
    was_scaled: bool


def build_frcrn_model(
    config: FRCRNConfig = FRCRNConfig(),
) -> Any:
    """Instantiate ClearerVoice/FRCRN after an explicit import guard."""
    try:
        from clearvoice import ClearVoice
    except Exception as exc:
        raise ClearVoiceDependencyError(
            "ClearerVoice is not importable. Install requirements-colab-frcrn.txt "
            "in a NumPy<2.0 environment; do not install it into the baseline/STFT "
            "environment."
        ) from exc

    try:
        return ClearVoice(
            task=config.task,
            model_names=list(config.model_names),
        )
    except Exception as exc:
        raise ClearVoiceDependencyError(
            "ClearerVoice was imported but FRCRN model initialization failed. "
            "Check the ClearerVoice version, checkpoint availability, and runtime."
        ) from exc


def inspect_model(model: Any) -> dict[str, Any]:
    """Return non-destructive model metadata without calling inference."""
    if model is None:
        raise ValueError("model must not be None.")
    info: dict[str, Any] = {
        "type": f"{type(model).__module__}.{type(model).__name__}",
        "mro": [cls.__name__ for cls in type(model).__mro__],
    }
    for attribute in ("task", "model_names", "models"):
        if hasattr(model, attribute):
            value = getattr(model, attribute)
            if attribute == "models" and isinstance(value, (list, tuple)):
                info[attribute] = [type(item).__name__ for item in value]
            else:
                info[attribute] = value
    callable_target = getattr(model, "__call__", None)
    if callable_target is not None:
        try:
            info["signature"] = str(inspect.signature(callable_target))
        except (TypeError, ValueError):
            info["signature"] = None
    return info


def normalize_waveform(value: Any, *, depth: int = 0) -> np.ndarray:
    """Normalize a ClearerVoice return value to a finite mono float32 waveform."""
    if depth > 6:
        raise TypeError("FRCRN result nesting is too deep to normalize.")
    if value is None:
        raise TypeError("FRCRN returned None.")

    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise TypeError(f"FRCRN returned a path that does not exist: {path}")
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ClearVoiceDependencyError(
                "soundfile is required to read a path returned by FRCRN."
            ) from exc
        audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
        return _to_mono_float32(audio)

    if isinstance(value, dict):
        for key in ("wav", "waveform", "audio", "enhanced", "output", "data"):
            if key in value:
                return normalize_waveform(value[key], depth=depth + 1)
        if not value:
            raise ValueError("FRCRN returned an empty dict.")
        return normalize_waveform(next(iter(value.values())), depth=depth + 1)

    if isinstance(value, (list, tuple)):
        candidates = [
            item
            for item in value
            if not isinstance(item, (str, int, float, np.integer, np.floating))
        ]
        if not candidates:
            raise TypeError("FRCRN returned a list/tuple without waveform data.")
        return normalize_waveform(candidates[0], depth=depth + 1)

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()

    array = np.asarray(value)
    if array.dtype == object:
        flat = array.ravel()
        if flat.size == 0:
            raise ValueError("FRCRN returned an empty object array.")
        return normalize_waveform(flat[0], depth=depth + 1)

    if np.iscomplexobj(array):
        array = np.real(array)
    if array.ndim == 0:
        raise ValueError("FRCRN returned a scalar instead of a waveform.")
    if array.ndim == 2:
        array = _to_mono_float32(array)
    if array.ndim != 1:
        raise ValueError(f"Could not normalize FRCRN output to mono; shape={array.shape}.")
    if array.size == 0:
        raise ValueError("FRCRN returned an empty waveform.")

    if array.dtype.kind in "iu":
        limits = np.iinfo(array.dtype)
        array = array.astype(np.float32) / float(max(abs(limits.min), limits.max))
    else:
        array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("FRCRN output contains non-finite samples.")
    return array


def _to_mono_float32(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        if array.shape[0] == 1:
            array = array.reshape(-1)
        elif array.shape[1] == 1:
            array = array.reshape(-1)
        elif array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
            array = array.mean(axis=0)
        elif array.shape[1] <= 8 and array.shape[0] > array.shape[1]:
            array = array.mean(axis=1)
        else:
            raise ValueError(f"Ambiguous FRCRN output shape: {array.shape}.")
    return array.astype(np.float32, copy=False)


def enhance_file(
    model: Any,
    path: str | Path,
    *,
    sample_rate: int = 16000,
) -> FRCRNResult:
    """Run FRCRN on one existing WAV file without writing through ClearerVoice."""
    if model is None:
        raise ValueError("FRCRN model is not initialized.")
    if int(sample_rate) != 16000:
        raise ValueError("FRCRN_SE_16K requires 16 kHz input.")
    audio_path = Path(path).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError(f"FRCRN input audio not found: {audio_path}")

    started = time.perf_counter()
    result = model(input_path=str(audio_path), online_write=False)
    elapsed = time.perf_counter() - started
    enhanced = normalize_waveform(result)

    peak = float(np.max(np.abs(enhanced)))
    was_scaled = peak >= 1.0
    if was_scaled:
        enhanced = (enhanced * (0.99 / peak)).astype(np.float32)
        peak = float(np.max(np.abs(enhanced)))

    model_names = getattr(model, "model_names", None)
    if isinstance(model_names, str):
        model_names = (model_names,)
    elif model_names is None:
        model_names = ("FRCRN_SE_16K",)
    else:
        model_names = tuple(str(name) for name in model_names)

    return FRCRNResult(
        enhanced=enhanced,
        elapsed_sec=float(elapsed),
        input_path=audio_path,
        model_names=model_names,
        output_peak=peak,
        was_scaled=bool(was_scaled),
    )


__all__ = [
    "ClearVoiceDependencyError",
    "FRCRNConfig",
    "FRCRNResult",
    "build_frcrn_model",
    "enhance_file",
    "inspect_model",
    "normalize_waveform",
]
