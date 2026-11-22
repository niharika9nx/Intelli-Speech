#!/usr/bin/env python3
"""Evaluate Wiener, oracle IRM, and the ZF-Net + GMM model on paired speech data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Allow `python scripts/evaluate_baselines.py` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from intelli_speech.audio_io import load_audio
from intelli_speech.baselines import BaselineResult, oracle_irm_enhance, wiener_enhance
from intelli_speech.dataset import VALID_SPLITS, load_pairs
from intelli_speech.evaluation import (
    MetricUnavailableError,
    evaluate_enhancement,
    optional_pesq,
    optional_stoi,
    summarize_metrics,
)
from intelli_speech.stft import STFTConfig, STFTDependencyError

DEFAULT_DATASET_DIR = Path(
    "/content/drive/MyDrive/Intelli-Speech-data/paired_dataset_v3"
)
METHODS = ("wiener", "oracle-irm", "zfnet-gmm")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
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
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="wiener",
        help="Method. oracle-irm uses the clean reference; zfnet-gmm needs checkpoints.",
    )
    parser.add_argument(
        "--gmm-checkpoint",
        type=Path,
        default=None,
        help="zfnet-gmm only: gmm.joblib from scripts/train_gmm.py.",
    )
    parser.add_argument(
        "--zfnet-checkpoint",
        type=Path,
        default=None,
        help="zfnet-gmm only: zfnet.pt from scripts/train_zfnet.py.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Evaluate only the first N metadata rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run output directory. Defaults to <dataset parent>/experiments/<method>_<split>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument("--n-fft", type=positive_int, default=512)
    parser.add_argument("--hop-length", type=positive_int, default=128)
    parser.add_argument(
        "--noise-duration",
        type=positive_float,
        default=0.25,
        help="Seconds from the noisy start used for the Wiener noise estimate.",
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


def evaluate(
    *,
    dataset_dir: Path,
    split: str,
    method: str,
    output_dir: Path,
    limit: int | None,
    overwrite: bool,
    n_fft: int,
    hop_length: int,
    noise_duration_sec: float,
    include_pesq: bool,
    include_stoi: bool,
    gmm_checkpoint: Path | None = None,
    zfnet_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run one baseline over one split and write metrics/config/summary files."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {VALID_SPLITS}.")

    dataset_path = Path(dataset_dir).expanduser().resolve()
    destination_path = Path(output_dir).expanduser().resolve()
    if destination_path == dataset_path or dataset_path in destination_path.parents:
        raise ValueError("Output directory must be outside the dataset directory.")

    pairs = load_pairs(dataset_path, split=split, limit=limit)
    if not pairs:
        raise ValueError(f"No pairs found for split {split!r}.")

    config = STFTConfig(
        n_fft=n_fft,
        hop_length=hop_length,
        sample_rate=16000,
    )
    enhancer = None
    if method == "zfnet-gmm":
        if gmm_checkpoint is None or zfnet_checkpoint is None:
            raise ValueError("zfnet-gmm requires --gmm-checkpoint and --zfnet-checkpoint.")
        from intelli_speech.enhancer import Enhancer

        enhancer = Enhancer.from_checkpoints(gmm_checkpoint, zfnet_checkpoint)
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

        predicted_noise = ""
        if method == "wiener":
            baseline = wiener_enhance(
                noisy,
                config,
                noise_duration_sec=noise_duration_sec,
            )
        elif method == "oracle-irm":
            baseline = oracle_irm_enhance(clean, noisy, config)
        else:
            detailed = enhancer.enhance_detailed(noisy)
            predicted_noise = detailed.output.noise_type
            baseline = BaselineResult(
                method="zfnet-gmm",
                enhanced=detailed.output.waveform,
                enhanced_stft=detailed.enhanced_stft,
                mask_or_gain=detailed.mask,
                uses_clean_reference=False,
            )

        metrics = evaluate_enhancement(
            clean,
            noisy,
            baseline.enhanced,
            sample_rate=16000,
        )
        row: dict[str, Any] = {
            "source_id": pair.source_id,
            "split": pair.split,
            "target_snr_db": pair.target_snr_db,
            "noise_environment": pair.noise_environment or "",
            "predicted_noise": predicted_noise,
            "method": baseline.method,
            "uses_clean_reference": baseline.uses_clean_reference,
            "noise_frames": baseline.noise_frames,
            "input_snr_db": metrics["input_snr_db"],
            "output_snr_db": metrics["output_snr_db"],
            "snr_improvement_db": metrics["snr_improvement_db"],
            "clean_rms": metrics["clean_rms"],
            "enhanced_rms": metrics["enhanced_rms"],
            "enhanced_peak": metrics["enhanced_peak"],
            "enhanced_clipping_detected": metrics["enhanced_clipping_detected"],
            "duration_sec": metrics["duration_sec"],
            "pesq": None,
            "stoi": None,
        }
        if include_pesq:
            row["pesq"] = optional_pesq(
                clean,
                baseline.enhanced,
                sample_rate=16000,
            )
        if include_stoi:
            row["stoi"] = optional_stoi(
                clean,
                baseline.enhanced,
                sample_rate=16000,
            )
        rows.append(row)
        print(
            f"[{index}/{len(pairs)}] {pair.source_id} "
            f"target={pair.target_snr_db:g} dB "
            f"input={row['input_snr_db']:.3f} dB "
            f"output={row['output_snr_db']:.3f} dB "
            f"change={row['snr_improvement_db']:+.3f} dB"
        )

    metric_names = [
        "input_snr_db",
        "output_snr_db",
        "snr_improvement_db",
        "clean_rms",
        "enhanced_rms",
        "enhanced_peak",
        "duration_sec",
    ]
    if include_pesq:
        metric_names.append("pesq")
    if include_stoi:
        metric_names.append("stoi")
    summary = summarize_metrics(rows, metric_names=metric_names)
    if method == "zfnet-gmm":
        labelled = [row for row in rows if row["noise_environment"]]
        if labelled:
            summary["noise_classification_accuracy"] = sum(
                row["predicted_noise"] == row["noise_environment"] for row in labelled
            ) / len(labelled)

    metrics_path = destination / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    config_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_path),
        "split": split,
        "method": method,
        "uses_clean_reference": method == "oracle-irm",
        "deployable": method != "oracle-irm",
        "limit": limit,
        "pair_count": len(rows),
        "stft": {
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "sample_rate": config.sample_rate,
            "window": config.window,
            "center": config.center,
        },
        "wiener_noise_duration_sec": noise_duration_sec if method == "wiener" else None,
        "include_pesq": include_pesq,
        "include_stoi": include_stoi,
        "gmm_checkpoint": str(gmm_checkpoint) if gmm_checkpoint else None,
        "zfnet_checkpoint": str(zfnet_checkpoint) if zfnet_checkpoint else None,
        "model_version": enhancer.model_version if enhancer else None,
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
        output_dir = (
            dataset_dir.parent
            / "experiments"
            / f"{args.method}_{args.split}"
        )

    try:
        result = evaluate(
            dataset_dir=dataset_dir,
            split=args.split,
            method=args.method,
            output_dir=output_dir,
            limit=args.limit,
            overwrite=args.overwrite,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            noise_duration_sec=args.noise_duration,
            include_pesq=args.include_pesq,
            include_stoi=args.include_stoi,
            gmm_checkpoint=args.gmm_checkpoint,
            zfnet_checkpoint=args.zfnet_checkpoint,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        ValueError,
        STFTDependencyError,
        MetricUnavailableError,
        RuntimeError,
    ) as exc:
        print(f"Baseline evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
