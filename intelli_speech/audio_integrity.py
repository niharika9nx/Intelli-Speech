"""Full-scale sample analysis: evidence for clipping versus simple peak normalisation.

Peak amplitude alone cannot distinguish the two. This module looks at *how* the
peak is reached:

- Peak normalisation scales a waveform so its largest sample sits at (or just below)
  full scale. Typically only one or a few isolated samples touch the maximum, and the
  amplitude histogram decays smoothly towards it.
- Clipping flattens the waveform at the rail. It produces runs of consecutive
  samples at (near) full scale and an excess ("spike") of samples in the extreme
  amplitude bin relative to neighbouring bins.

The classification is heuristic evidence, never proof. Thresholds are explicit
arguments so they can be reported and varied.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PCM16_POS_MAX = 32767
PCM16_NEG_MAX = -32768
NEAR_FULL_SCALE = 0.999


def _runs(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs in a boolean array."""
    if mask.size == 0 or not mask.any():
        return np.zeros(0, dtype=int)
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[1::2] - edges[0::2]


def full_scale_analysis(
    samples: np.ndarray,
    *,
    is_pcm16: bool,
    near_threshold: float = NEAR_FULL_SCALE,
    run_threshold: int = 3,
    spike_ratio_threshold: float = 5.0,
    histogram_bins: int = 1000,
) -> dict[str, Any]:
    """Describe full-scale behaviour of one mono recording.

    ``samples`` are int16 values when ``is_pcm16`` is True, otherwise floats in [-1, 1].
    """
    x = np.asarray(samples)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Expected a non-empty one-dimensional signal.")
    if is_pcm16:
        xi = x.astype(np.int32)
        exact = (xi >= PCM16_POS_MAX) | (xi <= PCM16_NEG_MAX)
        magnitude = np.abs(xi.astype(np.float64)) / 32768.0
    else:
        magnitude = np.abs(x.astype(np.float64))
        exact = magnitude >= 1.0
    near = magnitude >= near_threshold
    peak = float(magnitude.max())

    runs = _runs(near)
    exact_runs = _runs(exact)
    # Amplitude-histogram spike: top bin count vs mean of the 10 bins below it.
    hist, _ = np.histogram(magnitude, bins=histogram_bins, range=(0.0, max(peak, 1e-9)))
    neighbours = hist[-11:-1]
    neighbour_mean = float(neighbours.mean()) if neighbours.size else 0.0
    spike_ratio = float(hist[-1] / neighbour_mean) if neighbour_mean > 0 else (float("inf") if hist[-1] > 1 else 0.0)

    max_run = int(runs.max()) if runs.size else 0
    long_runs = int(np.sum(runs >= run_threshold))
    if max_run >= run_threshold or (spike_ratio >= spike_ratio_threshold and hist[-1] >= 3):
        verdict = "clipping_evidence"
    elif int(exact.sum()) <= 2 and max_run <= 2 and peak >= near_threshold:
        verdict = "consistent_with_peak_normalisation"
    elif peak < near_threshold:
        verdict = "below_full_scale"
    else:
        verdict = "indeterminate"

    return {
        "n_samples": int(x.size),
        "peak": round(peak, 6),
        "exact_full_scale_samples": int(exact.sum()),
        "near_full_scale_samples": int(near.sum()),
        "near_full_scale_fraction": float(near.sum() / x.size),
        "near_runs": int(runs.size),
        "max_consecutive_near_full_scale": max_run,
        "max_consecutive_exact_full_scale": int(exact_runs.max()) if exact_runs.size else 0,
        f"runs_ge_{run_threshold}": long_runs,
        "top_bin_spike_ratio": round(spike_ratio, 3) if np.isfinite(spike_ratio) else None,
        "verdict": verdict,
        "thresholds": {
            "near": near_threshold,
            "run": run_threshold,
            "spike_ratio": spike_ratio_threshold,
        },
    }
