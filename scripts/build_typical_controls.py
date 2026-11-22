#!/usr/bin/env python3
"""Build the bounded, speaker-balanced typical Telugu control set (typical_controls_v2).

Two explicit phases (run in Colab after `notebook_login()` and accepting the
ARTPARK-IISc/Vaani access conditions):

  --phase plan      metadata-only scan + deterministic selection; writes
                    selection_plan.csv and plan.json (counts, estimated bytes). No audio.
  --phase download  reads plan, fetches ONLY the selected rows' audio, writes
                    audio/<speakerID>/ctl_XXXX.wav (16 kHz mono PCM16), manifest.csv,
                    provenance.json. Refuses to run if manifest.csv already exists.

Output: <data-root>/typical_controls_v2/ (new folder; existing Drive data untouched).
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("plan", "download"), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-shards", type=int, default=30, help="Evenly spread Telugu shards scanned for metadata.")
    parser.add_argument("--n-speakers", type=int, default=40)
    parser.add_argument("--max-per-speaker", type=int, default=10)
    parser.add_argument("--max-total", type=int, default=400)
    parser.add_argument("--min-per-speaker", type=int, default=3)
    parser.add_argument("--overwrite-plan", action="store_true")
    return parser


def default_fs():
    from huggingface_hub import HfFileSystem

    return HfFileSystem()  # uses the token cached by notebook_login(); never printed


def source_revision() -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(VAANI_REPO).sha
    except Exception:
        return None


def run(args: argparse.Namespace, fs=None) -> dict[str, Any]:
    data_root = args.data_root.expanduser()
    output = args.output_dir or data_root / "typical_controls_v2"
    for protected in (data_root / "vaani_telugu_transcribed", data_root / "paired_dataset_v3"):
        output = ensure_outside_dataset(protected, output)
    fs = fs or default_fs()
    plan_csv, plan_json = output / "selection_plan.csv", output / "plan.json"

    if args.phase == "plan":
        refuse_overwrite(plan_json, overwrite=args.overwrite_plan)
        shards_all = list_telugu_shards(fs)
        if not shards_all:
            raise FileNotFoundError("No Telugu parquet shards found; check access conditions and repo layout.")
        shards = spread_subset(shards_all, args.max_shards)
        metadata = scan_metadata(fs, shards)
        config = SelectionConfig(
            n_speakers=args.n_speakers,
            max_per_speaker=args.max_per_speaker,
            max_total=args.max_total,
            min_per_speaker=args.min_per_speaker,
        )
        selected, report = select_controls(metadata, config)
        selected.to_csv(plan_csv, index=False)
        plan = {
            "created_utc": utc_now(),
            "source_repo": VAANI_REPO,
            "source_revision": source_revision(),
            "telugu_shards_total": len(shards_all),
            "shards_scanned": len(shards),
            "estimated_audio_bytes": estimate_audio_bytes(fs, selected) if len(selected) else 0,
            **report,
        }
        write_json(plan_json, plan)
        return plan

    # download phase
    if not plan_csv.is_file():
        raise FileNotFoundError("selection_plan.csv missing; run --phase plan first.")
    manifest_path = refuse_overwrite(output / "manifest.csv", overwrite=False)
    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    selected = pd.read_csv(plan_csv, dtype={"speakerID": str, "_utt": str})
    manifest = fetch_selected(fs, selected, output, source_revision=plan.get("source_revision"))
    manifest.to_csv(manifest_path, index=False)
    ok = manifest[manifest["status"] == "ok"]
    stats = {
        "created_utc": utc_now(),
        "source_repo": VAANI_REPO,
        "source_revision": plan.get("source_revision"),
        "license": "CC-BY-4.0 (gated; access conditions accepted by the downloading user)",
        "rows_planned": int(len(selected)),
        "files_written": int(len(ok)),
        "duplicates_skipped": int((manifest["status"] == "duplicate_skipped").sum()),
        "decode_errors": int((manifest["status"] == "decode_error").sum()),
        "speakers": int(ok["speakerID"].nunique()) if len(ok) else 0,
        "clips_per_speaker": ok.groupby("speakerID").size().describe().round(2).to_dict() if len(ok) else {},
        "total_minutes": round(float(ok["duration_s"].sum()) / 60, 2) if len(ok) else 0.0,
        "resampled_files": int(ok["resampled"].sum()) if len(ok) else 0,
        "original_sample_rates": ok["original_sample_rate"].value_counts().to_dict() if len(ok) else {},
    }
    write_json(output / "provenance.json", stats)
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Typical-control build failed: {exc}", file=sys.stderr)
        return 1
    safe = {k: v for k, v in result.items() if k not in ("selected_speaker_ids",)}
    print(json.dumps(safe, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
