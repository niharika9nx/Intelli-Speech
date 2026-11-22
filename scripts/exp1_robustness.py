#!/usr/bin/env python3
"""Experiment 1 robustness checks for the pattern catalogue.

1. Control replication: the same speaker-level analysis against the secondary control
   set (``vaani_copy``) instead of ``controls_v2``.
2. Duration matching: atypical recordings (~35 s) are much longer than control clips
   (~6 s). Each atypical recording is cut into consecutive windows of the median control
   duration; temporal + prosodic features are recomputed per window (same per-recording
   VAD as the controls) and the analysis is rerun against ``controls_v2``.
3. Summary: a feature is ``robust_class1`` only if it is ``candidate_class1`` in the main
   catalogue and in both checks.

Reads features.csv (Drive) and the atypical corpus (HF cache). Writes
<features dir>/robustness/. No transcript text is read or written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd

from intelli_speech import acoustic_features as af
from intelli_speech.experiment import refuse_overwrite, utc_now, write_json
from scripts.analyze_exp1 import analyse

SR = 16000


def windows(x: np.ndarray, window_s: float, sr: int = SR, min_fraction: float = 0.5) -> Iterator[tuple[int, np.ndarray]]:
    """Consecutive non-overlapping windows; a short tail is kept if >= min_fraction of a window."""
    size = int(round(window_s * sr))
    for k, start in enumerate(range(0, x.size, size)):
        chunk = x[start:start + size]
        if chunk.size >= max(int(min_fraction * size), int(0.5 * sr)):
            yield k, chunk


def window_features(records: Iterator[dict[str, Any]], window_s: float) -> pd.DataFrame:
    from intelli_speech.vaani_controls import to_mono_16k

    rows = []
    for item in records:
        mono = to_mono_16k(item["audio"], item["rate"])
        for k, chunk in windows(mono, window_s):
            row = {"dataset": "atypical_windows", "group": "atypical", "speaker_id": item["speaker_id"],
                   "record_id": f"{item['record_id']}_w{k:03d}", "source_record_id": item["record_id"]}
            try:
                row.update(af.extract_recording_features(chunk, SR, include_voice=False, include_spectral=False))
            except Exception as exc:  # keep going; report per window
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return pd.DataFrame(rows)


def compare(catalogues: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for name, cat in catalogues.items():
        part = cat[["feature", "family", "verdict", "consistency_k", "cliffs_delta"]].rename(
            columns={"verdict": f"verdict_{name}", "consistency_k": f"k_{name}", "cliffs_delta": f"delta_{name}"})
        merged = part if merged is None else merged.merge(part.drop(columns="family"), on="feature", how="outer")
    verdict_cols = [f"verdict_{n}" for n in catalogues]
    merged["robust_class1"] = (merged[verdict_cols] == "candidate_class1").all(axis=1)
    return merged.sort_values(["robust_class1", "feature"], ascending=[False, True]).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <features dir>/robustness")
    parser.add_argument("--window-s", type=float, default=None, help="Default: median controls_v2 duration")
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--skip-duration-matched", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace, records: Iterator[dict[str, Any]] | None = None) -> dict[str, Any]:
    features = pd.read_csv(args.features, dtype={"speaker_id": str})
    output = args.output_dir or args.features.parent / "robustness"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = refuse_overwrite(output / "robustness_summary.csv", overwrite=args.overwrite)

    main_path = args.features.parent / "analysis" / "catalogue.csv"
    main_cat = pd.read_csv(main_path) if main_path.is_file() else analyse(
        features, control_dataset="controls_v2", n_perm=args.permutations)[0]
    catalogues = {"main": main_cat}
    info: dict[str, Any] = {"created_utc": utc_now(), "features": str(args.features)}

    if (features["dataset"] == "vaani_copy").any():
        rep, _ = analyse(features, control_dataset="vaani_copy", n_perm=args.permutations)
        rep.to_csv(output / "catalogue_vs_vaani_copy.csv", index=False)
        catalogues["vs_vaani_copy"] = rep
    else:
        info["vs_vaani_copy"] = "skipped: no vaani_copy rows"

    if not args.skip_duration_matched:
        controls = features[features["dataset"] == "controls_v2"]
        window_s = args.window_s or float(controls["duration_s"].median())
        if records is None:
            from scripts.extract_features_exp1 import iter_atypical
            records = iter_atypical(args.hf_cache)
        win = window_features(records, window_s)
        win.to_csv(output / "atypical_window_features.csv", index=False)
        combined = pd.concat([win, controls], ignore_index=True)
        dm, _ = analyse(combined, control_dataset="controls_v2", n_perm=args.permutations)
        dm.to_csv(output / "catalogue_duration_matched.csv", index=False)
        catalogues["duration_matched"] = dm
        ok = win[win["error"].isna()] if "error" in win else win
        info["duration_matched"] = {
            "window_s": round(window_s, 3),
            "windows": int(len(win)), "windows_ok": int(len(ok)),
            "windows_per_speaker": ok.groupby("speaker_id").size().describe().round(2).to_dict(),
        }

    summary = compare(catalogues)
    summary.to_csv(summary_path, index=False)
    info["checks"] = list(catalogues)
    info["robust_class1"] = summary.loc[summary["robust_class1"], "feature"].tolist()
    info["verdict_counts"] = {n: c["verdict"].value_counts().to_dict() for n, c in catalogues.items()}
    write_json(output / "robustness.json", info)
    return info


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        info = run(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Robustness check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
