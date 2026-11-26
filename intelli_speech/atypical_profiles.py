"""Per-fold timing profiles for synthetic atypical speech (Phase 3).

Leave-one-speaker-out: for the fold that holds out atypical speaker H, profiles are
fitted only from the *other* atypical speakers (never H), relative to the typical
control speakers (``controls_v2``). One profile per training atypical speaker:

  stretch            = control median articulation rate / speaker articulation rate (>= 1)
  pause_rate_per_min = speaker median pause rate (pauses >= 250 ms)
  pause_mu, sigma    = log-normal fitted from speaker medians of the pause median and
                       P90 (pauses >= 250 ms): mu = ln(median), sigma = ln(P90/median)/1.2816

Speaker values are medians over that speaker's recordings (Exp 1 features.csv), the same
unit of inference as Exp 1. A synthetic speaker samples one profile, a degree
``alpha`` in [alpha_min, 1] that interpolates control → profile (alpha is the
"how atypical" label for the GMM), and small multiplicative jitter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .temporal_transform import TemporalParams

PROFILE_FEATURES = ("articulation_rate_nuclei_per_s", "pause_rate_per_min_250ms", "pause_median_s_250ms", "pause_p90_s_250ms")
Z90 = 1.2816


@dataclass(frozen=True)
class Profile:
    source_speaker: str          # pseudonym of the atypical speaker the profile came from
    stretch: float
    pause_rate_per_min: float
    pause_mu: float
    pause_sigma: float


def _speaker_medians(features: pd.DataFrame, group: str, dataset: str | None = None) -> pd.DataFrame:
    rows = features[features["group"] == group]
    if dataset is not None:
        rows = rows[rows["dataset"] == dataset]
    if "error" in rows:
        rows = rows[rows["error"].isna()]
    return rows.groupby("speaker_id")[list(PROFILE_FEATURES)].median()


def _lognormal(median: float, p90: float) -> tuple[float, float]:
    median = max(float(median), 0.26)
    p90 = max(float(p90), median * 1.05)
    return float(np.log(median)), float(max(0.1, np.log(p90 / median) / Z90))


def control_profile(features: pd.DataFrame, control_dataset: str = "controls_v2") -> Profile:
    c = _speaker_medians(features, "typical", control_dataset).median()
    mu, sigma = _lognormal(c["pause_median_s_250ms"], c["pause_p90_s_250ms"])
    return Profile("control", 1.0, float(c["pause_rate_per_min_250ms"]), mu, sigma)


def fit_fold(features: pd.DataFrame, held_out: str, *, control_dataset: str = "controls_v2",
             pseudonyms: dict[str, str] | None = None) -> dict[str, Any]:
    """Profiles for the fold that holds out atypical speaker ``held_out``."""
    atyp = _speaker_medians(features, "atypical")
    if held_out not in atyp.index:
        raise KeyError("held-out speaker not in features")
    control = control_profile(features, control_dataset)
    ctrl_rate = float(_speaker_medians(features, "typical", control_dataset)["articulation_rate_nuclei_per_s"].median())
    names = pseudonyms or {s: s for s in atyp.index}
    profiles = []
    for spk, row in atyp.drop(index=held_out).iterrows():
        if row.isna().any():
            continue
        mu, sigma = _lognormal(row["pause_median_s_250ms"], row["pause_p90_s_250ms"])
        profiles.append(Profile(names.get(spk, spk), float(max(1.0, ctrl_rate / row["articulation_rate_nuclei_per_s"])),
                                float(row["pause_rate_per_min_250ms"]), mu, sigma))
    return {"held_out": names.get(held_out, held_out), "control": asdict(control),
            "control_articulation_rate": ctrl_rate, "profiles": [asdict(p) for p in profiles]}


def sample_params(fold: dict[str, Any], rng: np.random.Generator, *, alpha_min: float = 0.3, jitter: float = 0.10,
                  seed: int | None = None) -> tuple[TemporalParams, dict[str, Any]]:
    """Draw one synthetic speaker's TemporalParams from a fold; returns (params, meta)."""
    profiles = fold["profiles"]
    if not profiles:
        raise ValueError("fold has no profiles")
    k = int(rng.integers(len(profiles)))
    p, c = profiles[k], fold["control"]
    alpha = float(rng.uniform(alpha_min, 1.0))
    j = lambda: float(np.exp(rng.normal(0.0, jitter)))  # noqa: E731
    stretch = max(1.0, (1.0 + alpha * (p["stretch"] - 1.0)) * j())
    rate = max(0.0, (c["pause_rate_per_min"] + alpha * (p["pause_rate_per_min"] - c["pause_rate_per_min"])) * j())
    mu = c["pause_mu"] + alpha * (p["pause_mu"] - c["pause_mu"]) + float(np.log(j()))
    sigma = max(0.1, (c["pause_sigma"] + alpha * (p["pause_sigma"] - c["pause_sigma"])) * j())
    params = TemporalParams(stretch=stretch, pause_rate_per_min=rate, pause_mu=mu, pause_sigma=sigma,
                            seed=int(seed if seed is not None else rng.integers(2**31)))
    return params, {"profile_index": k, "profile_source": p["source_speaker"], "alpha": round(alpha, 4)}
