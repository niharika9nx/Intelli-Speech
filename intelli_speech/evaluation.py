from __future__ import annotations

import statistics
from typing import Iterable, Mapping, Sequence

import numpy as np

from .metrics import clipping_report, peak_amplitude, rms_amplitude


class MetricUnavailableError(RuntimeError):
    """Raised when an optional objective metric package is unavailable."""


def calculate_snr_db(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    epsilon: float = 1e-20,
) -> float:
    """Compute an error-to-reference SNR in dB for aligned waveforms.

    This is the score used by the archived IRM experiment. It treats
    ``reference - estimate`` as the error and is not a replacement for a
    perceptual metric.
    """
    if float(epsilon) <= 0:
        raise ValueError("epsilon must be positive.")
    reference_array = np.asarray(reference, dtype=np.float64)
    estimate_array = np.asarray(estimate, dtype=np.float64)
    if reference_array.shape != estimate_array.shape:
        raise ValueError(
            f"Reference/estimate shapes differ: "
            f"{reference_array.shape} vs {estimate_array.shape}"
        )
    if reference_array.size == 0:
        raise ValueError("Reference and estimate are empty.")
    if not np.isfinite(reference_array).all() or not np.isfinite(estimate_array).all():
        raise ValueError("Reference or estimate contains non-finite samples.")

    signal_power = float(np.mean(reference_array**2))
    error_power = float(np.mean((reference_array - estimate_array) ** 2))
    if signal_power <= 0:
        raise ValueError("Reference signal has zero power.")
    return float(10.0 * np.log10(signal_power / max(error_power, float(epsilon))))


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Alias for :func:`calculate_snr_db` with the default epsilon."""
    return calculate_snr_db(reference, estimate)


def snr_improvement_db(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
) -> float:
    """Return output SNR minus input SNR on the same clean reference."""
    input_score = calculate_snr_db(clean, noisy)
    output_score = calculate_snr_db(clean, enhanced)
    return float(output_score - input_score)


def evaluate_enhancement(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    *,
    sample_rate: int | None = None,
) -> dict[str, float | int]:
    """Compute the first objective metrics for one clean/noisy/enhanced triplet."""
    clean_array = np.asarray(clean, dtype=np.float32)
    noisy_array = np.asarray(noisy, dtype=np.float32)
    enhanced_array = np.asarray(enhanced, dtype=np.float32)
    if not (clean_array.shape == noisy_array.shape == enhanced_array.shape):
        raise ValueError(
            "Clean, noisy, and enhanced waveforms must have equal shapes; "
            f"received {clean_array.shape}, {noisy_array.shape}, "
            f"and {enhanced_array.shape}."
        )
    if clean_array.ndim != 1 or clean_array.size == 0:
        raise ValueError("Expected non-empty one-dimensional waveforms.")
    if not (
        np.isfinite(clean_array).all()
        and np.isfinite(noisy_array).all()
        and np.isfinite(enhanced_array).all()
    ):
        raise ValueError("One or more waveforms contain non-finite samples.")

    input_snr = calculate_snr_db(clean_array, noisy_array)
    output_snr = calculate_snr_db(clean_array, enhanced_array)
    clipping = clipping_report(enhanced_array)
    metrics: dict[str, float | int] = {
        "input_snr_db": input_snr,
        "output_snr_db": output_snr,
        "snr_improvement_db": float(output_snr - input_snr),
        "clean_rms": rms_amplitude(clean_array),
        "noisy_rms": rms_amplitude(noisy_array),
        "enhanced_rms": rms_amplitude(enhanced_array),
        "enhanced_peak": peak_amplitude(enhanced_array),
        "enhanced_full_scale_samples": int(clipping["full_scale_sample_count"]),
        "enhanced_clipping_detected": int(bool(clipping["clipping_detected"])),
        "length_samples": int(clean_array.size),
    }
    if sample_rate is not None:
        if int(sample_rate) <= 0:
            raise ValueError("sample_rate must be positive.")
        metrics["duration_sec"] = float(clean_array.size / int(sample_rate))
    return metrics


def summarize_metrics(
    rows: Sequence[Mapping[str, object]],
    metric_names: Iterable[str] | None = None,
) -> dict[str, object]:
    """Summarize numeric per-file metrics across a run."""
    if not rows:
        raise ValueError("Cannot summarize an empty result set.")

    if metric_names is None:
        candidates = {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        names = sorted(candidates)
    else:
        names = list(metric_names)

    summaries: dict[str, dict[str, float | int]] = {}
    for name in names:
        values = []
        for row in rows:
            value = row.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Metric {name!r} is not numeric in every row.")
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"Metric {name!r} contains a non-finite value.")
            values.append(number)
        if not values:
            raise ValueError(f"Metric {name!r} has no values.")
        summaries[name] = {
            "count": len(values),
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        }

    return {"count": len(rows), "metrics": summaries}


def optional_pesq(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int = 16000,
    mode: str | None = None,
) -> float:
    """Compute PESQ when the optional ``pesq`` package is installed."""
    try:
        from pesq import pesq
    except ImportError as exc:
        raise MetricUnavailableError(
            "PESQ requires the optional 'pesq' package."
        ) from exc

    reference_array = np.asarray(reference, dtype=np.float32)
    estimate_array = np.asarray(estimate, dtype=np.float32)
    if reference_array.shape != estimate_array.shape:
        raise ValueError("PESQ inputs must have equal shapes.")
    if int(sample_rate) not in (8000, 16000):
        raise ValueError("PESQ supports 8 kHz or 16 kHz audio.")
    selected_mode = mode or ("wb" if int(sample_rate) == 16000 else "nb")
    return float(pesq(int(sample_rate), reference_array, estimate_array, selected_mode))


def optional_stoi(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int = 16000,
    extended: bool = False,
) -> float:
    """Compute STOI when the optional ``pystoi`` package is installed."""
    try:
        from pystoi import stoi
    except ImportError as exc:
        raise MetricUnavailableError(
            "STOI requires the optional 'pystoi' package."
        ) from exc

    reference_array = np.asarray(reference, dtype=np.float32)
    estimate_array = np.asarray(estimate, dtype=np.float32)
    if reference_array.shape != estimate_array.shape:
        raise ValueError("STOI inputs must have equal shapes.")
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive.")
    return float(
        stoi(
            reference_array,
            estimate_array,
            int(sample_rate),
            extended=bool(extended),
        )
    )


__all__ = [
    "MetricUnavailableError",
    "calculate_snr_db",
    "evaluate_enhancement",
    "optional_pesq",
    "optional_stoi",
    "snr_db",
    "snr_improvement_db",
    "summarize_metrics",
]
