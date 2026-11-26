#!/usr/bin/env python3
"""Closed-loop validation of the DSP temporal module on real typical source clips.

For each leave-one-speaker-out fold (held-out atypical speaker H):
  1. fit timing profiles from the other atypical speakers (Exp 1 features.csv);
  2. calibration: transform training clips with each profile at full degree (alpha=1,
     no jitter), re-measure with the Exp 1 extractor, compare the synthetic speaker
     median with the real profile speaker's median;
  3. coverage: transform clips with randomly sampled synthetic speakers (alpha in
     [0.3, 1], jitter) and check whether H's real median lies inside the synthetic
     speakers' P5–P95 range (can a model trained on this fold see something like H?);
  4. pitch preservation: output/input median F0 ratio.

Inputs are read-only: Exp 1 features.csv and <source>/quality.csv (usable train clips).
Outputs: <data-root>/analysis_v2/temporal_validation_<date>/ (folds/*.json with
pseudonymous speakers S1..S7, CSV tables, summary.json, a few example WAVs).
Real atypical speaker IDs are never written.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd

from intelli_speech import acoustic_features as af
from intelli_speech.atypical_profiles import fit_fold, sample_params
from intelli_speech.experiment import refuse_overwrite, utc_now, write_json
from intelli_speech.temporal_transform import TRANSFORM_VERSION, TemporalParams, click_score, transform

DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/Intelli-Speech-data")
SR = 16000
# Exp 1 robust class-1 features (main + vs_vaani_copy + duration-matched), 2026-11-22.
ROBUST_TIMING_FEATURES = (
    "articulation_rate_nuclei_per_s", "speech_rate_nuclei_per_s", "speech_activity_ratio",
    "pause_rate_per_min_250ms", "pause_rate_per_min_500ms",
    "pause_p90_s_150ms", "pause_p90_s_250ms", "pause_p90_s_500ms", "pause_median_s_500ms",
    "pause_iqr_s_150ms", "pause_iqr_s_250ms",
    "pause_time_fraction_150ms", "pause_time_fraction_250ms", "pause_time_fraction_500ms",
)
MEASURED = (*ROBUST_TIMING_FEATURES, "f0_median_hz")


def pseudonyms(features: pd.DataFrame) -> dict[str, str]:
    ids = sorted(features.loc[features["group"] == "atypical", "speaker_id"].astype(str).unique())
    return {s: f"S{i + 1}" for i, s in enumerate(ids)}


def measure(x: np.ndarray) -> dict[str, Any]:
    f = af.extract_recording_features(x, SR, include_voice=False, include_spectral=False)
    return {k: f.get(k) for k in MEASURED}


def load_train_clips(source_dir: Path) -> pd.DataFrame:
    quality = pd.read_csv(source_dir / "quality.csv", dtype={"speakerID": str})
    manifest = pd.read_csv(source_dir / "manifest.csv", dtype={"speakerID": str})
    clips = quality[quality["usable"] & (quality["split"] == "train")].merge(manifest[["clip_id", "file"]], on="clip_id")
    return clips.sort_values("clip_id").reset_index(drop=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import soundfile as sf

    data_root = args.data_root.expanduser()
    output = args.output_dir or data_root / "analysis_v2" / f"temporal_validation_{TRANSFORM_VERSION}_{date.today().isoformat()}"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = refuse_overwrite(output / "summary.json", overwrite=args.overwrite)
    features = pd.read_csv(args.features, dtype={"speaker_id": str})
    alias = pseudonyms(features)
    usable = features[features["error"].isna()] if "error" in features else features
    atyp = usable[usable["group"] == "atypical"]
    real = atyp.groupby("speaker_id")[list(ROBUST_TIMING_FEATURES)].median().rename(index=alias)
    controls = usable[(usable["group"] == "typical") & (usable["dataset"] == "controls_v2")]
    ctrl = controls.groupby("speaker_id")[list(ROBUST_TIMING_FEATURES)].median()
    ctrl_lo, ctrl_hi, ctrl_med = ctrl.quantile(0.10), ctrl.quantile(0.90), ctrl.median()

    clips = load_train_clips(args.source_dir)
    if len(clips) < args.clips_per_fold:
        raise ValueError("Not enough usable train clips in the source set.")
    (output / "folds").mkdir(exist_ok=True)
    (output / "examples").mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    rng_master = np.random.default_rng(args.seed)
    for fold_index, (held_real, held) in enumerate(sorted(alias.items(), key=lambda kv: kv[1])):
        fold = fit_fold(features, held_real, pseudonyms=alias)
        write_json(output / "folds" / f"fold_{held}.json", fold)
        rng = np.random.default_rng(rng_master.integers(2**31))
        picks = clips.iloc[rng.choice(len(clips), args.clips_per_fold, replace=False)]
        speakers: dict[str, Any] = {}
        for n, (_, clip) in enumerate(picks.iterrows()):
            audio, rate = sf.read(str(args.source_dir / clip["file"]), dtype="float32")
            mode = "calibration" if n < args.clips_per_fold // 2 else "coverage"
            if mode == "calibration":
                k = n % len(fold["profiles"])
                p = fold["profiles"][k]
                params = TemporalParams(stretch=p["stretch"], pause_rate_per_min=p["pause_rate_per_min"],
                                        pause_mu=p["pause_mu"], pause_sigma=p["pause_sigma"], seed=int(rng.integers(2**31)))
                meta = {"profile_source": p["source_speaker"], "alpha": 1.0, "synthetic_speaker": f"{held}-cal-{p['source_speaker']}"}
            else:
                speaker = f"{held}-cov-{(n - args.clips_per_fold // 2) // args.clips_per_speaker}"
                if speaker not in speakers:  # one parameter draw per synthetic speaker, shared by its clips
                    speakers[speaker] = sample_params(fold, rng)
                base, meta = speakers[speaker]
                params = TemporalParams(**{**base.__dict__, "seed": int(rng.integers(2**31))})
                meta = {**meta, "synthetic_speaker": speaker}
            row: dict[str, Any] = {"fold": held, "mode": mode, "clip_id": clip["clip_id"], **meta,
                                   "stretch": params.stretch, "pause_rate_target": params.pause_rate_per_min}
            try:
                y, labels, info = transform(audio, rate, params)
                before, after = measure(audio), measure(y)
                row.update({f"out_{k}": v for k, v in after.items()})
                row["f0_ratio"] = (after["f0_median_hz"] / before["f0_median_hz"]) if before["f0_median_hz"] and after["f0_median_hz"] else None
                row.update({"click_in": click_score(audio, rate), "click_out": click_score(y, rate), "splices": info["splices"],
                            "inserted_pauses": info["inserted_pauses"], "insertion_shortfall": info["insertion_shortfall"],
                            "output_s": info["output_s"], "input_s": info["input_s"]})
                if n < args.examples_per_fold:  # listening set: isolate which component sounds wrong
                    stem = output / "examples" / f"{held}_{n:02d}"
                    sf.write(f"{stem}_0_input.wav", audio, rate, subtype="PCM_16")
                    only_stretch = TemporalParams(stretch=params.stretch, pause_rate_per_min=0.0, pause_mu=float(np.log(0.26)),
                                                  pause_sigma=0.1, seed=params.seed)
                    only_pauses = TemporalParams(**{**params.__dict__, "stretch": 1.0})
                    sf.write(f"{stem}_1_stretch_only.wav", transform(audio, rate, only_stretch)[0], rate, subtype="PCM_16")
                    sf.write(f"{stem}_2_pauses_only.wav", transform(audio, rate, only_pauses)[0], rate, subtype="PCM_16")
                    sf.write(f"{stem}_3_both.wav", y, rate, subtype="PCM_16")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        print(f"[fold {held}] {fold_index + 1}/{len(alias)} done ({len(fold['profiles'])} profiles)", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(output / "validation_clips.csv", index=False)
    ok = table[table["error"].isna()] if "error" in table else table

    # calibration: synthetic speaker (fold x profile) median vs the real profile speaker
    cal = ok[ok["mode"] == "calibration"].groupby(["fold", "profile_source"])[[f"out_{f}" for f in ROBUST_TIMING_FEATURES]].median()
    cal_rows = []
    for (fold_name, src), med in cal.iterrows():
        for f in ROBUST_TIMING_FEATURES:
            target, value = real.loc[src, f], med[f"out_{f}"]
            direction = np.sign(real[f].median() - ctrl_med[f])
            outside = (value > ctrl_hi[f]) if direction > 0 else (value < ctrl_lo[f])
            cal_rows.append({"fold": fold_name, "profile_source": src, "feature": f, "synthetic": value, "real_target": target,
                             "rel_error": (value - target) / abs(target) if target else None,
                             "outside_control_band_atypical_direction": bool(outside) if pd.notna(value) else None})
    calibration = pd.DataFrame(cal_rows)
    calibration.to_csv(output / "calibration.csv", index=False)

    # coverage: held-out real speaker inside synthetic speakers' P5–P95?
    cov = ok[ok["mode"] == "coverage"].groupby(["fold", "synthetic_speaker"])[[f"out_{f}" for f in ROBUST_TIMING_FEATURES]].median()
    cov_rows = []
    for fold_name, grp in cov.groupby(level="fold"):
        for f in ROBUST_TIMING_FEATURES:
            lo, hi = grp[f"out_{f}"].quantile(0.05), grp[f"out_{f}"].quantile(0.95)
            value = real.loc[fold_name, f]
            cov_rows.append({"fold": fold_name, "feature": f, "held_out_real": value, "synthetic_p5": lo, "synthetic_p95": hi,
                             "covered": bool(lo <= value <= hi) if pd.notna(value) and pd.notna(lo) else None})
    coverage = pd.DataFrame(cov_rows)
    coverage.to_csv(output / "coverage.csv", index=False)

    summary = {
        "created_utc": utc_now(), "transform_version": TRANSFORM_VERSION,
        "features": str(args.features), "source_dir": str(args.source_dir),
        "clips_per_fold": args.clips_per_fold, "folds": len(alias), "transforms": int(len(table)),
        "errors": int(table["error"].notna().sum()) if "error" in table else 0,
        "f0_ratio_quantiles": ok["f0_ratio"].quantile([0.05, 0.5, 0.95]).round(4).tolist() if "f0_ratio" in ok else None,
        "insertion_shortfall_total": int(ok["insertion_shortfall"].sum()) if "insertion_shortfall" in ok else None,
        "click_score_in_quantiles": ok["click_in"].quantile([0.5, 0.9, 0.99]).round(2).tolist() if "click_in" in ok else None,
        "click_score_out_quantiles": ok["click_out"].quantile([0.5, 0.9, 0.99]).round(2).tolist() if "click_out" in ok else None,
        "click_increase_gt_1p5x_fraction": round(float((ok["click_out"] > 1.5 * ok["click_in"]).mean()), 3) if "click_out" in ok else None,
        "calibration_median_abs_rel_error_by_feature": calibration.assign(a=calibration["rel_error"].abs())
            .groupby("feature")["a"].median().round(3).to_dict(),
        "calibration_outside_control_band_rate": round(float(calibration["outside_control_band_atypical_direction"].mean()), 3),
        "coverage_rate_by_feature": coverage.groupby("feature")["covered"].mean().round(3).to_dict(),
        "coverage_rate_by_fold": coverage.groupby("fold")["covered"].mean().round(3).to_dict(),
        "coverage_rate_overall": round(float(coverage["covered"].mean()), 3),
        "note": "Validation of the DSP module only; not an intelligibility claim.",
    }
    write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True, help="Exp 1 features.csv")
    parser.add_argument("--source-dir", type=Path, required=True, help="synth_source_v1_full")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--clips-per-fold", type=int, default=120)
    parser.add_argument("--clips-per-speaker", type=int, default=3, help="Coverage clips per synthetic speaker.")
    parser.add_argument("--examples-per-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (FileExistsError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Temporal validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
