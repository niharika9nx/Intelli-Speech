#!/usr/bin/env python3
"""Experiment 1 analysis — speaker-level pattern catalogue from features.csv.

Unit of inference = speaker (each speaker's median per feature, equal weight).
For every feature:
  * within-atypical ICC(1) (between-speaker share of variance),
  * atypical vs control speaker medians: Cliff's delta, Hedges g, speaker-level
    permutation p, bootstrap CI (over speakers) of the median difference,
  * consistency k/N: atypical speakers outside the control P10–P90 band, same direction,
  * leave-one-atypical-speaker-out robustness and small-speaker-exclusion sensitivity,
  * channel-confound check (Spearman |rho| with bandwidth / rms within groups),
  * Benjamini–Hochberg FDR across features.
Verdicts (evidence classes for the synthetic-data catalogue):
  candidate_class1 | speaker_specific | channel_confounded | not_supported | insufficient_data
These are statistical observations about 7 speakers, not impairment markers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd

from intelli_speech.experiment import refuse_overwrite, utc_now, write_json
from intelli_speech.grouped_stats import cliffs_delta

ID_COLUMNS = {"dataset", "group", "record_id", "speaker_id", "difficulty_level", "prompt_file_type", "gender",
              "district", "error", "feature_version", "full_scale_verdict"}
CHANNEL_COLUMNS = ("effective_bandwidth_hz", "rms_dbfs", "near_full_scale_fraction")
EXCLUDED_FEATURES = {"duration_s", "nuclei_count", "active_duration_s", "speech_time_s", "phonation_time_s"}
FAMILIES = (("pause", "temporal"), ("speech_", "temporal"), ("articulation", "temporal"), ("internucleus", "temporal"),
            ("f0_", "prosodic"), ("voiced", "prosodic"), ("intensity", "prosodic"), ("energy_mod", "prosodic"),
            ("nucleus_prom", "prosodic"), ("hnr", "voice"), ("cpps", "voice"), ("jitter", "voice"), ("shimmer", "voice"),
            ("spectral", "spectral"), ("ltas", "spectral"), ("mfcc", "spectral"))


def family(feature: str) -> str:
    for prefix, name in FAMILIES:
        if feature.startswith(prefix):
            return name
    return "other"


def icc1(values: pd.Series, groups: pd.Series) -> float | None:
    """One-way random-effects ICC(1) from recordings nested in speakers."""
    frame = pd.DataFrame({"v": values, "g": groups}).dropna()
    if frame["g"].nunique() < 2:
        return None
    k = frame.groupby("g").size()
    n, a = len(frame), k.size
    grand = frame["v"].mean()
    ssb = (k * (frame.groupby("g")["v"].mean() - grand) ** 2).sum()
    ssw = ((frame["v"] - frame.groupby("g")["v"].transform("mean")) ** 2).sum()
    if n - a <= 0:
        return None
    msb, msw = ssb / (a - 1), ssw / (n - a)
    k0 = (n - (k**2).sum() / n) / (a - 1)
    denom = msb + (k0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else None


def hedges_g(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    if pooled == 0:
        return None
    correction = 1 - 3 / (4 * (a.size + b.size) - 9)
    return float((a.mean() - b.mean()) / pooled * correction)


def permutation_p(a: np.ndarray, b: np.ndarray, n: int, rng: np.random.Generator) -> float:
    observed = abs(np.median(a) - np.median(b))
    pooled = np.concatenate([a, b])
    hits = 0
    for _ in range(n):
        perm = rng.permutation(pooled)
        if abs(np.median(perm[: a.size]) - np.median(perm[a.size:])) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    diffs = [np.median(rng.choice(a, a.size)) - np.median(rng.choice(b, b.size)) for _ in range(n)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def consistency(atyp: pd.Series, control: np.ndarray) -> tuple[int, int, str]:
    lo, hi = np.percentile(control, 10), np.percentile(control, 90)
    above, below = int((atyp > hi).sum()), int((atyp < lo).sum())
    return (above, int(atyp.size), "higher") if above >= below else (below, int(atyp.size), "lower")


def bh_fdr(pvalues: list[float | None]) -> list[float | None]:
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    if not idx:
        return [None] * len(pvalues)
    p = np.array([pvalues[i] for i in idx])
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out: list[float | None] = [None] * len(pvalues)
    for rank, i in enumerate(order):
        out[idx[i]] = float(min(q[rank], 1.0))
    return out


def analyse(features: pd.DataFrame, *, control_dataset: str, min_speaker_recordings: int = 10, n_perm: int = 5000,
            n_boot: int = 2000, seed: int = 0, consistency_k: int = 5, fdr_q: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    data = features[features.get("error").isna()] if "error" in features else features
    data = data[(data["group"] == "atypical") | (data["dataset"] == control_dataset)]
    numeric = [c for c in data.columns if c not in ID_COLUMNS and c not in EXCLUDED_FEATURES
               and c not in CHANNEL_COLUMNS and pd.api.types.is_numeric_dtype(data[c])]
    speaker = data.groupby(["group", "speaker_id"])[numeric + list(c for c in CHANNEL_COLUMNS if c in data)].median()
    speaker["n_recordings"] = data.groupby(["group", "speaker_id"]).size()
    atyp_rec = data[data["group"] == "atypical"]
    small = set(speaker.loc["atypical"].index[speaker.loc["atypical", "n_recordings"] < min_speaker_recordings]) if "atypical" in speaker.index.get_level_values(0) else set()

    rows = []
    for feature in numeric:
        a = speaker.loc["atypical", feature].dropna() if "atypical" in speaker.index.get_level_values(0) else pd.Series(dtype=float)
        c = speaker.loc["typical", feature].dropna().to_numpy() if "typical" in speaker.index.get_level_values(0) else np.array([])
        row: dict[str, Any] = {"feature": feature, "family": family(feature), "n_atypical_speakers": int(a.size), "n_control_speakers": int(c.size)}
        if a.size < 4 or c.size < 5:
            row["verdict"] = "insufficient_data"
            rows.append(row)
            continue
        av = a.to_numpy()
        k, n, direction = consistency(a, c)
        loso_signs, loso_k = [], []
        for spk in a.index:
            rest = a.drop(spk)
            d = cliffs_delta(rest.tolist(), c.tolist())
            loso_signs.append(np.sign(d) if d else 0.0)
            loso_k.append(consistency(rest, c)[0])
        excl = a.drop([s for s in a.index if s in small], errors="ignore")
        delta = cliffs_delta(av.tolist(), c.tolist())
        lo, hi = bootstrap_ci(av, c, n_boot, rng)
        rho = {}
        for ch in CHANNEL_COLUMNS:
            if ch in data and ch != feature:
                for grp, sub in data.groupby("group"):
                    r = sub[[feature, ch]].dropna()
                    if len(r) >= 10:
                        rho[f"rho_{ch}_{grp}"] = float(r[feature].rank().corr(r[ch].rank()))
        row.update({
            "icc1_atypical": icc1(atyp_rec[feature], atyp_rec["speaker_id"]),
            "atypical_median_of_speakers": float(np.median(av)),
            "control_median_of_speakers": float(np.median(c)),
            "cliffs_delta": delta,
            "hedges_g": hedges_g(av, c),
            "median_diff_ci95": [round(lo, 6), round(hi, 6)],
            "perm_p": permutation_p(av, c, n_perm, rng),
            "consistency_k": k,
            "consistency_n": n,
            "direction": direction,
            "loso_sign_stable": bool(len(set(loso_signs)) == 1 and loso_signs[0] != 0),
            "loso_min_k": int(min(loso_k)),
            "delta_excluding_small_speakers": cliffs_delta(excl.tolist(), c.tolist()) if excl.size >= 3 else None,
            "max_abs_channel_rho": max((abs(v) for v in rho.values()), default=None),
            **rho,
        })
        rows.append(row)

    catalogue = pd.DataFrame(rows)
    if "perm_p" in catalogue:
        catalogue["fdr_q"] = bh_fdr([None if pd.isna(p) else float(p) for p in catalogue["perm_p"]])
    verdicts = []
    for _, r in catalogue.iterrows():
        if r.get("verdict") == "insufficient_data":
            verdicts.append("insufficient_data")
        elif r.get("max_abs_channel_rho") is not None and r["max_abs_channel_rho"] >= 0.5:
            verdicts.append("channel_confounded")
        elif (r["consistency_k"] >= consistency_k and r["loso_sign_stable"] and r["fdr_q"] is not None
              and r["fdr_q"] < fdr_q and abs(r["cliffs_delta"] or 0) >= 0.474):
            verdicts.append("candidate_class1")
        elif 1 <= r["consistency_k"] <= 2:
            verdicts.append("speaker_specific")
        else:
            verdicts.append("not_supported")
    catalogue["verdict"] = verdicts
    return catalogue, speaker.reset_index()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <features dir>/analysis")
    parser.add_argument("--control-dataset", default="auto", choices=("auto", "controls_v2", "vaani_copy"))
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features = pd.read_csv(args.features, dtype={"speaker_id": str})
    control = args.control_dataset
    if control == "auto":
        control = "controls_v2" if (features["dataset"] == "controls_v2").any() else "vaani_copy"
    output = args.output_dir or args.features.parent / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    try:
        catalogue_path = refuse_overwrite(output / "catalogue.csv", overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"Analysis refused: {exc}", file=sys.stderr)
        return 1
    catalogue, speakers = analyse(features, control_dataset=control, n_perm=args.permutations)
    catalogue.to_csv(catalogue_path, index=False)
    speakers.to_csv(output / "speaker_summary.csv", index=False)
    summary = {
        "created_utc": utc_now(),
        "control_dataset": control,
        "verdict_counts": catalogue["verdict"].value_counts().to_dict(),
        "criteria": "candidate_class1: >=5/7 speakers outside control P10-P90 same direction, LOSO sign-stable, "
                    "BH q<0.10, |Cliff's delta|>=0.474 (large), max |channel rho|<0.5",
        "note": "Observations about 7 speakers; not clinical markers.",
    }
    write_json(output / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
