#!/usr/bin/env python3
"""Run the ClearerVoice FRCRN adapter on one or more paired test samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Allow `python scripts/run_frcrn.py` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from intelli_speech.audio_io import load_audio
from intelli_speech.dataset import VALID_SPLITS, load_pairs
from intelli_speech.evaluation import (
    MetricUnavailableError,
    evaluate_enhancement,
    optional_pesq,
    optional_stoi,
    summarize_metrics,
)
from intelli_speech.frcrn import (
    ClearVoiceDependencyError,
    FRCRNConfig,
    build_frcrn_model,
    enhance_file,
)
from intelli_speech.preprocessing import write_pcm16

DEFAULT_DATASET_DIR = Path(
    "/content/drive/MyDrive/Intelli-Speech-data/paired_dataset_v3"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Paired dataset directory containing metadata.csv (read-only).",
    )
    parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="test",
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=1,
        help="Process only the first N metadata rows (default 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <dataset parent>/experiments/clearvoice_frcrn.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing enhanced WAV files and result files.",
    )
    parser.add_argument(
        "--model-name",
        default="FRCRN_SE_16K",
        help="ClearerVoice model name.",
    )
    parser.add_argument(
        "--include-pesq",
        action="store_true",
        help="Compute PESQ when the optional pesq package is installed.",
    )
    parser.add_argument(
        "--include-stoi",
        action="store_true",
        help="Compute STOI when the optional pystoi package is installed.",
    )
    return parser


def prepare_output_dir(path: Path, *, overwrite: bool) -> Path:
    output_dir = path.expanduser()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new directory or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _unique_output_path(directory: Path, source_id: str) -> Path:
    candidate = directory / f"{source_id}_frcrn.wav"
    suffix = 2
    while candidate.exists():
        candidate = directory / f"{source_id}_frcrn_{suffix}.wav"
        suffix += 1
    return candidate


def evaluate(
    *,
    dataset_dir: Path,
    split: str,
    output_dir: Path,
    limit: int,
    overwrite: bool,
    model_name: str,
    include_pesq: bool,
    include_stoi: bool,
) -> dict[str, Any]:
    """Run FRCRN on a bounded number of pairs and write metrics/audio outputs."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {VALID_SPLITS}.")
    dataset_path = Path(dataset_dir).expanduser().resolve()
    destination_path = Path(output_dir).expanduser().resolve()
    if destination_path == dataset_path or dataset_path in destination_path.parents:
        raise ValueError("Output directory must be outside the dataset directory.")

    pairs = load_pairs(dataset_path, split=split, limit=limit)
    if not pairs:
        raise ValueError(f"No pairs found for split {split!r}.")

    config = FRCRNConfig(model_names=(model_name,))
    model = build_frcrn_model(config)
    destination = prepare_output_dir(destination_path, overwrite=overwrite)

    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        clean, clean_rate = load_audio(pair.clean_path)
        noisy, noisy_rate = load_audio(pair.noisy_path)
        if clean_rate != 16000 or noisy_rate != 16000:
            raise ValueError(
                f"Expected 16 kHz audio for {pair.source_id}; "
                f"received {clean_rate} and {noisy_rate}."
            )
        if clean.shape != noisy.shape:
            raise ValueError(
                f"Clean/noisy lengths differ for {pair.source_id}: "
                f"{clean.shape} vs {noisy.shape}."
            )

        result = enhance_file(model, pair.noisy_path, sample_rate=16000)
        if result.enhanced.shape != noisy.shape:
            raise ValueError(
                f"FRCRN output length differs for {pair.source_id}: "
                f"input={noisy.shape}, output={result.enhanced.shape}."
            )

        audio_path = (
            destination / f"{pair.source_id}_frcrn.wav"
            if overwrite
            else _unique_output_path(destination, pair.source_id)
        )
        write_pcm16(audio_path, result.enhanced, sample_rate=16000)

        metrics = evaluate_enhancement(
            clean,
            noisy,
            result.enhanced,
            sample_rate=16000,
        )
        row: dict[str, Any] = {
            "source_id": pair.source_id,
            "split": pair.split,
            "target_snr_db": pair.target_snr_db,
            "noise_environment": pair.noise_environment or "",
            "method": "frcrn",
            "model_name": model_name,
            "uses_clean_reference": False,
            "elapsed_sec": result.elapsed_sec,
            "output_peak": result.output_peak,
            "output_was_scaled": result.was_scaled,
            "audio_file": audio_path.name,
            "input_snr_db": metrics["input_snr_db"],
            "output_snr_db": metrics["output_snr_db"],
            "snr_improvement_db": metrics["snr_improvement_db"],
            "clean_rms": metrics["clean_rms"],
            "enhanced_rms": metrics["enhanced_rms"],
            "enhanced_peak": metrics["enhanced_peak"],
            "duration_sec": metrics["duration_sec"],
            "pesq": None,
            "stoi": None,
        }
        if include_pesq:
            row["pesq"] = optional_pesq(
                clean,
                result.enhanced,
                sample_rate=16000,
            )
        if include_stoi:
            row["stoi"] = optional_stoi(
                clean,
                result.enhanced,
                sample_rate=16000,
            )
        rows.append(row)
        print(
            f"[{index}/{len(pairs)}] {pair.source_id} "
            f"target={pair.target_snr_db:g} dB "
            f"input={row['input_snr_db']:.3f} dB "
            f"output={row['output_snr_db']:.3f} dB "
            f"change={row['snr_improvement_db']:+.3f} dB "
            f"time={row['elapsed_sec']:.2f}s"
        )

    metric_names = [
        "input_snr_db",
        "output_snr_db",
        "snr_improvement_db",
        "elapsed_sec",
        "output_peak",
        "duration_sec",
    ]
    if include_pesq:
        metric_names.append("pesq")
    if include_stoi:
        metric_names.append("stoi")
    summary = summarize_metrics(rows, metric_names=metric_names)

    metrics_path = destination / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    config_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_path),
        "split": split,
        "method": "frcrn",
        "model_name": model_name,
        "uses_clean_reference": False,
        "deployable": True,
        "limit": limit,
        "pair_count": len(rows),
        "include_pesq": include_pesq,
        "include_stoi": include_stoi,
    }
    (destination / "config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )
    (destination / "summary.json").write_text(
        json.dumps(
            {
                "config": config_payload,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "output_dir": str(destination),
        "metrics_csv": str(metrics_path),
        "summary_json": str(destination / "summary.json"),
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = args.dataset_dir.expanduser()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = dataset_dir.parent / "experiments" / "clearvoice_frcrn"

    try:
        result = evaluate(
            dataset_dir=dataset_dir,
            split=args.split,
            output_dir=output_dir,
            limit=args.limit,
            overwrite=args.overwrite,
            model_name=args.model_name,
            include_pesq=args.include_pesq,
            include_stoi=args.include_stoi,
        )
    except (
        ClearVoiceDependencyError,
        FileExistsError,
        FileNotFoundError,
        MetricUnavailableError,
        ValueError,
    ) as exc:
        print(f"FRCRN evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
