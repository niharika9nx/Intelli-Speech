#!/usr/bin/env python3
"""Train the GMM noise-type classifier on noisy training mixtures.

Writes ``gmm.joblib`` and ``gmm_report.json`` (validation accuracy, confusion matrix,
calibrated temperature) into the run directory. The dataset is only read.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np

from intelli_speech.audio_io import load_audio
from intelli_speech.dataset import AudioPair, load_pairs
from intelli_speech.experiment import (
    DEFAULT_DATASET_DIR,
    default_run_dir,
    ensure_outside_dataset,
    refuse_overwrite,
    utc_now,
    write_json,
)
from intelli_speech.features import DEFAULT_N_MELS, log_mel_frames
from intelli_speech.gmm_noise import GMMDependencyError, NoiseGMM
from intelli_speech.stft import STFTConfig, STFTDependencyError, compute_stft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS)
    parser.add_argument("--frames-per-file", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N train rows.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def utterance_features(pair: AudioPair, config: STFTConfig, n_mels: int) -> np.ndarray:
    noisy, _ = load_audio(pair.noisy_path)
    return log_mel_frames(compute_stft(noisy, config), config, n_mels=n_mels)


def labelled(pairs: list[AudioPair], split: str) -> list[AudioPair]:
    missing = [p.source_id for p in pairs if not p.noise_environment]
    if missing:
        raise ValueError(
            f"{len(missing)} {split} rows have no noise_environment label "
            f"(e.g. {missing[:3]}); the GMM needs it."
        )
    return pairs


def train(args: argparse.Namespace) -> dict:
    dataset_dir = args.dataset_dir.expanduser()
    run_dir = ensure_outside_dataset(
        dataset_dir, args.output_dir or default_run_dir(dataset_dir)
    )
    gmm_path = refuse_overwrite(run_dir / "gmm.joblib", overwrite=args.overwrite)
    config = STFTConfig()
    rng = np.random.default_rng(args.seed)

    train_pairs = labelled(load_pairs(dataset_dir, split="train", limit=args.limit), "train")
    val_pairs = labelled(load_pairs(dataset_dir, split="validation"), "validation")

    started = time.time()
    by_class: dict[str, list[np.ndarray]] = defaultdict(list)
    for index, pair in enumerate(train_pairs, start=1):
        frames = utterance_features(pair, config, args.n_mels)
        if frames.shape[0] > args.frames_per_file:
            frames = frames[rng.choice(frames.shape[0], args.frames_per_file, replace=False)]
        by_class[str(pair.noise_environment)].append(frames)
        if index % 100 == 0 or index == len(train_pairs):
            print(f"[features] {index}/{len(train_pairs)} train files")

    gmm = NoiseGMM(n_components=args.n_components, random_state=args.seed)
    print(f"[fit] {len(by_class)} classes: {sorted(by_class)}")
    gmm.fit(by_class)

    val_features = [utterance_features(p, config, args.n_mels) for p in val_pairs]
    val_labels = [str(p.noise_environment) for p in val_pairs]
    unknown = sorted(set(val_labels) - set(gmm.classes))
    if unknown:
        raise ValueError(f"Validation labels not seen in training: {unknown}")
    temperature = gmm.calibrate_temperature(val_features, val_labels)

    predictions = [gmm.predict(item) for item in val_features]
    accuracy = float(np.mean([p == t for p, t in zip(predictions, val_labels)]))
    confusion = {
        truth: {pred: 0 for pred in gmm.classes} for truth in gmm.classes
    }
    for truth, pred in zip(val_labels, predictions):
        confusion[truth][pred] += 1
    by_snr: dict[str, list[bool]] = defaultdict(list)
    for pair, pred in zip(val_pairs, predictions):
        by_snr[f"{pair.target_snr_db:g}dB"].append(pred == pair.noise_environment)

    gmm.save(gmm_path)
    report = {
        "created_utc": utc_now(),
        "dataset_dir": str(dataset_dir),
        "train_files": len(train_pairs),
        "validation_files": len(val_pairs),
        "classes": gmm.classes,
        "n_components": args.n_components,
        "n_mels": args.n_mels,
        "feature_dim": gmm.feature_dim,
        "frames_per_file": args.frames_per_file,
        "temperature": temperature,
        "validation_accuracy": accuracy,
        "validation_accuracy_by_snr": {k: float(np.mean(v)) for k, v in by_snr.items()},
        "validation_confusion": confusion,
        "seconds": round(time.time() - started, 1),
        "gmm_path": str(gmm_path),
    }
    write_json(run_dir / "gmm_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = train(args)
    except (FileExistsError, FileNotFoundError, ValueError, STFTDependencyError, GMMDependencyError) as exc:
        print(f"GMM training failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"GMM validation accuracy: {report['validation_accuracy']:.3f} "
        f"(temperature {report['temperature']:.3f}); saved {report['gmm_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
