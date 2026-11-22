"""Speaker-aware statistics: speakers, not clips, are the independent units.

Used by the follow-up audit (difficulty-label associations) and planned for
Experiment 1 (equal-weight speaker summaries, within-speaker tests).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def speaker_equal_weight_mean(values: Sequence[float], groups: Sequence[str]) -> float | None:
    """Mean of per-speaker means (each speaker weighted equally)."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    keep = np.isfinite(v)
    if not keep.any():
        return None
    means = [v[keep & (g == s)].mean() for s in np.unique(g[keep])]
    return float(np.mean(means))


def stratified_permutation_test(
    values: Sequence[float],
    labels: Sequence[str],
    groups: Sequence[str],
    *,
    positive: str,
    negative: str,
    n_permutations: int = 5000,
    min_per_cell: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Within-group test of ``median(positive) - median(negative)``.

    Only groups (speakers) that contain at least ``min_per_cell`` clips of *both*
    labels contribute. The statistic is the equal-weight mean of per-group median
    differences; labels are permuted *within* each group, so between-speaker
    differences cannot create a significant result.
    """
    v = np.asarray(values, dtype=float)
    lab = np.asarray(labels).astype(str)
    grp = np.asarray(groups).astype(str)
    ok = np.isfinite(v) & np.isin(lab, [positive, negative])
    v, lab, grp = v[ok], lab[ok], grp[ok]

    eligible = []
    for s in np.unique(grp):
        m = grp == s
        if (lab[m] == positive).sum() >= min_per_cell and (lab[m] == negative).sum() >= min_per_cell:
            eligible.append(s)
    if not eligible:
        return {"eligible_groups": 0, "statistic": None, "p_value": None, "per_group": {}}

    def statistic(labels_array: np.ndarray) -> tuple[float, dict[str, float]]:
        diffs = {}
        for s in eligible:
            m = grp == s
            diffs[s] = float(np.median(v[m & (labels_array == positive)]) - np.median(v[m & (labels_array == negative)]))
        return float(np.mean(list(diffs.values()))), diffs

    observed, per_group = statistic(lab)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_permutations):
        permuted = lab.copy()
        for s in eligible:
            idx = np.flatnonzero(grp == s)
            permuted[idx] = rng.permutation(permuted[idx])
        if abs(statistic(permuted)[0]) >= abs(observed) - 1e-12:
            exceed += 1
    signs = [np.sign(d) for d in per_group.values()]
    return {
        "eligible_groups": len(eligible),
        "statistic": observed,
        "p_value": (exceed + 1) / (n_permutations + 1),
        "per_group": per_group,
        "sign_consistency": float(max(signs.count(1.0), signs.count(-1.0)) / len(signs)),
        "positive": positive,
        "negative": negative,
    }


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Cliff's delta effect size: P(a > b) - P(a < b)."""
    x = np.asarray([v for v in a if np.isfinite(v)], dtype=float)
    y = np.asarray([v for v in b if np.isfinite(v)], dtype=float)
    if x.size == 0 or y.size == 0:
        return None
    greater = (x[:, None] > y[None, :]).sum()
    less = (x[:, None] < y[None, :]).sum()
    return float((greater - less) / (x.size * y.size))
