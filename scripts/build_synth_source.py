#!/usr/bin/env python3
"""Build the typical-Telugu *source* set for synthetic atypical data (Phase 3).

Source: regular Vaani Telugu (``ARTPARK-IISc/Vaani``, gated, CC-BY-4.0), spontaneous
image-prompted speech, same collection pipeline as the atypical corpus.

Phases (run in Colab after ``notebook_login()``):
  --phase plan      metadata-only scan; excludes every speaker already used as a
                    control (typical_controls_v2, vaani_telugu_transcribed); speaker-
                    balanced deterministic selection; transcripts NOT required
                    (transcribed clips are flagged). Writes selection_plan.csv, plan.json.
  --phase download  fetches only the selected rows' audio → audio/<speakerID>/src_XXXX.wav
                    (16 kHz mono PCM16), manifest.csv, provenance.json.
  --phase quality   screens every clip (SNR estimate, bandwidth, clipping, speech time,
                    activity); keeps speakers with >= --min-pass-per-speaker passing clips;
                    assigns a deterministic speaker-level train/val/test split.
                    Writes quality.csv, splits.csv, quality_summary.json. Files are never
                    deleted; rejected clips are only marked.

Output: <data-root>/synth_source_v1_<set>/ (new folder; nothing existing is modified).
Every phase refuses to overwrite its own outputs.
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

import pandas as pd

from intelli_speech.experiment import ensure_outside_dataset, refuse_overwrite, utc_now, write_json
from intelli_speech.source_quality import QualityConfig, assign_speaker_splits, clip_quality, config_dict, failing_reasons
from intelli_speech.vaani_controls import (
    VAANI_REPO,
    SelectionConfig,
    estimate_audio_bytes,
    fetch_selected,
    list_telugu_shards,
    scan_metadata,
    select_controls,
    spread_subset,
)

DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/Intelli-Speech-data")
SALT = "intelli-speech-synth-source-v1"
PRESETS = {
    # ~30 % oversampling over the clip target to absorb the quality filter.
    "pilot": {"n_speakers": 65, "max_per_speaker": 20, "max_total": 1300, "min_per_speaker": 10, "max_shards": 30},
    "full": {"n_speakers": 430, "max_per_speaker": 20, "max_total": 8600, "min_per_speaker": 10, "max_shards": 80},
}
PROTECTED = ("vaani_telugu_transcribed", "paired_dataset_v3", "typical_controls_v2", "openslr_slr66")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("plan", "download", "quality"), required=True)
    parser.add_argument("--set", dest="set_name", choices=tuple(PRESETS), default="pilot")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    for key in ("n_speakers", "max_per_speaker", "max_total", "min_per_speaker", "max_shards"):
        parser.add_argument(f"--{key.replace('_', '-')}", type=int, default=None, help="Override the preset.")
    parser.add_argument("--min-pass-per-speaker", type=int, default=5)
    parser.add_argument("--overwrite-plan", action="store_true")
    return parser


def default_fs():
    from huggingface_hub import HfFileSystem

    return HfFileSystem()  # token cached by notebook_login(); never printed


def source_revision() -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(VAANI_REPO).sha
    except Exception:
        return None


def excluded_speakers(data_root: Path) -> tuple[set[str], dict[str, Any]]:
    """Speakers already used as controls; they must never appear in the synthetic source."""
    excluded: set[str] = set()
    report: dict[str, Any] = {}
    manifest = data_root / "typical_controls_v2" / "manifest.csv"
    if manifest.is_file():
        ids = set(pd.read_csv(manifest, dtype={"speakerID": str})["speakerID"].dropna().astype(str).str.strip())
        excluded |= ids
        report["typical_controls_v2"] = len(ids)
    else:
        report["typical_controls_v2"] = "missing"
    copy_dir = data_root / "vaani_telugu_transcribed"
    if (copy_dir / "metadata.csv").is_file():
        from intelli_speech.vaani import load_typical_vaani_metadata

        ids = set(load_typical_vaani_metadata(copy_dir)["speakerID"].dropna().astype(str).str.strip())
        excluded |= ids
        report["vaani_telugu_transcribed"] = len(ids)
    else:
        report["vaani_telugu_transcribed"] = "missing"
    report["total_excluded_speakers"] = len(excluded)
    return excluded, report


def _settings(args: argparse.Namespace) -> dict[str, int]:
    preset = dict(PRESETS[args.set_name])
    for key in preset:
        if getattr(args, key, None) is not None:
            preset[key] = getattr(args, key)
    return preset


def plan_phase(args, fs, output: Path, data_root: Path) -> dict[str, Any]:
    refuse_overwrite(output / "plan.json", overwrite=args.overwrite_plan)
    settings = _settings(args)
    excluded, exclusion_report = excluded_speakers(data_root)
    if "missing" in exclusion_report.values():
        raise FileNotFoundError(f"Control speaker lists missing ({exclusion_report}); cannot guarantee disjoint speakers.")
    shards_all = list_telugu_shards(fs)
    if not shards_all:
        raise FileNotFoundError("No Telugu parquet shards found; check access conditions and repo layout.")
    shards = spread_subset(shards_all, settings["max_shards"])
    metadata = scan_metadata(fs, shards)
    ids = metadata["speakerID"].astype(str).str.strip()
    removed = int(ids.isin(excluded).sum())
    metadata = metadata[~ids.isin(excluded)]
    config = SelectionConfig(n_speakers=settings["n_speakers"], max_per_speaker=settings["max_per_speaker"],
                             max_total=settings["max_total"], min_per_speaker=settings["min_per_speaker"],
                             require_transcript=False, balance_gender=True, salt=SALT)
    selected, report = select_controls(metadata, config)
    text = selected.get("transcript", pd.Series(index=selected.index, dtype=object)).fillna("").astype(str).str.strip()
    selected["has_transcript"] = text != ""
    selected.to_csv(output / "selection_plan.csv", index=False)
    plan = {
        "created_utc": utc_now(), "set": args.set_name, "settings": settings,
        "source_repo": VAANI_REPO, "source_revision": source_revision(),
        "telugu_shards_total": len(shards_all), "shards_scanned": len(shards),
        "exclusions": exclusion_report, "rows_removed_by_exclusion": removed,
        "selected_with_transcript": int(selected["has_transcript"].sum()),
        "estimated_audio_bytes": estimate_audio_bytes(fs, selected) if len(selected) else 0,
        **report,
    }
    write_json(output / "plan.json", plan)
    return plan


def download_phase(fs, output: Path) -> dict[str, Any]:
    plan_csv = output / "selection_plan.csv"
    if not plan_csv.is_file():
        raise FileNotFoundError("selection_plan.csv missing; run --phase plan first.")
    manifest_path = refuse_overwrite(output / "manifest.csv", overwrite=False)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(plan_csv, dtype={"speakerID": str, "_utt": str})
    manifest = fetch_selected(fs, selected, output, source_revision=plan.get("source_revision"), id_prefix="src")
    manifest = manifest.rename(columns={"control_id": "clip_id"})
    manifest.to_csv(manifest_path, index=False)
    ok = manifest[manifest["status"] == "ok"]
    stats = {
        "created_utc": utc_now(), "set": plan.get("set"), "source_repo": VAANI_REPO,
        "source_revision": plan.get("source_revision"),
        "license": "CC-BY-4.0 (gated; access conditions accepted by the downloading user)",
        "rows_planned": int(len(selected)), "files_written": int(len(ok)),
        "duplicates_skipped": int((manifest["status"] == "duplicate_skipped").sum()),
        "decode_errors": int((manifest["status"] == "decode_error").sum()),
        "speakers": int(ok["speakerID"].nunique()) if len(ok) else 0,
        "total_minutes": round(float(ok["duration_s"].sum()) / 60, 2) if len(ok) else 0.0,
        "original_sample_rates": ok["original_sample_rate"].value_counts().to_dict() if len(ok) else {},
    }
    write_json(output / "provenance.json", stats)
    return stats


def quality_phase(args, output: Path, config: QualityConfig = QualityConfig()) -> dict[str, Any]:
    import soundfile as sf

    manifest_path = output / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("manifest.csv missing; run --phase download first.")
    quality_path = refuse_overwrite(output / "quality.csv", overwrite=False)
    manifest = pd.read_csv(manifest_path, dtype={"speakerID": str})
    rows = []
    for _, item in manifest[manifest["status"] == "ok"].iterrows():
        row: dict[str, Any] = {"clip_id": item["clip_id"], "speakerID": item["speakerID"], "gender": item.get("gender"),
                               "district": item.get("district"), "duration_s": item.get("duration_s"),
                               "has_transcript": item.get("has_transcript")}
        try:
            audio, rate = sf.read(str(output / item["file"]), dtype="float32")
            q = clip_quality(audio, rate)
            reasons = failing_reasons(q, config)
            row.update(q)
            row.update({"quality_pass": not reasons, "reject_reasons": ";".join(reasons)})
        except Exception as exc:
            row.update({"quality_pass": False, "reject_reasons": f"read_error:{type(exc).__name__}"})
        rows.append(row)
    quality = pd.DataFrame(rows)

    passing = quality[quality["quality_pass"]]
    per_speaker = passing.groupby("speakerID").agg(n_pass=("clip_id", "size"),
                                                   gender=("gender", lambda g: g.mode().iloc[0] if g.notna().any() else None))
    kept = per_speaker[per_speaker["n_pass"] >= args.min_pass_per_speaker].reset_index()
    kept["split"] = assign_speaker_splits(kept, salt=SALT).to_numpy()
    quality["split"] = quality["speakerID"].map(dict(zip(kept["speakerID"], kept["split"])))
    quality.loc[~quality["quality_pass"], "split"] = None
    quality["usable"] = quality["split"].notna()
    quality.to_csv(quality_path, index=False)
    kept.to_csv(output / "splits.csv", index=False)

    reasons = quality.loc[~quality["quality_pass"], "reject_reasons"].str.split(";").explode().value_counts().to_dict()
    usable = quality[quality["usable"]]
    summary = {
        "created_utc": utc_now(), "thresholds": config_dict(config), "min_pass_per_speaker": args.min_pass_per_speaker,
        "clips_screened": int(len(quality)), "clips_pass": int(quality["quality_pass"].sum()),
        "pass_rate": round(float(quality["quality_pass"].mean()), 4) if len(quality) else None,
        "reject_reasons": reasons,
        "speakers_screened": int(quality["speakerID"].nunique()),
        "speakers_kept": int(len(kept)),
        "usable_clips": int(len(usable)),
        "usable_minutes": round(float(usable["duration_s"].sum()) / 60, 2),
        "by_split": {s: {"speakers": int(g["speakerID"].nunique()), "clips": int(len(g)),
                         "minutes": round(float(g["duration_s"].sum()) / 60, 2),
                         "gender_speakers": g.drop_duplicates("speakerID")["gender"].astype(str).value_counts().to_dict()}
                     for s, g in usable.groupby("split")},
        "usable_with_transcript": int(usable["has_transcript"].fillna(False).astype(bool).sum()),
        "metric_medians_all": {c: float(quality[c].median()) for c in ("snr_db_estimate", "effective_bandwidth_hz", "speech_activity_ratio") if c in quality},
    }
    write_json(output / "quality_summary.json", summary)
    return summary


def run(args: argparse.Namespace, fs=None) -> dict[str, Any]:
    data_root = args.data_root.expanduser()
    output = args.output_dir or data_root / f"synth_source_v1_{args.set_name}"
    for name in PROTECTED:
        output = ensure_outside_dataset(data_root / name, output)
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "plan":
        return plan_phase(args, fs or default_fs(), output, data_root)
    if args.phase == "download":
        return download_phase(fs or default_fs(), output)
    return quality_phase(args, output)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Synthetic-source build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
