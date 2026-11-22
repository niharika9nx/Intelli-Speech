#!/usr/bin/env python3
"""Train the GMM-conditioned ZF-Net ratio-mask estimator.

Requires a trained ``gmm.joblib`` (scripts/train_gmm.py). Target: the ideal ratio mask
``|S| / (|S| + |N|)`` computed from the clean target and the noise residual. Writes
``zfnet.pt`` (best validation loss), ``history.csv``, ``train_config.json`` and
``train_summary.json`` into the run directory. The dataset is only read.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
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
from intelli_speech.features import log_mel_frames
from intelli_speech.gmm_noise import NoiseGMM
from intelli_speech.stft import STFTConfig, compute_stft
from intelli_speech.zfnet import (
    ZFNetConfig,
    ZFNetMask,
    count_parameters,
    import_torch,
    mask_loss,
    save_zfnet_checkpoint,
    torch_istft,
    torch_normalized_log_magnitude,
    torch_stft,
)

MODEL_VERSION = "zfnet-gmm-v1"
SAMPLE_RATE = 16000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gmm", type=Path, default=None, help="Defaults to <output-dir>/gmm.joblib.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-sec", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="Smoke-test cap.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N train rows.")
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda.")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_split(pairs: list[AudioPair], gmm: NoiseGMM, config: STFTConfig):
    """Load waveforms and per-utterance GMM posteriors into memory."""
    items = []
    for index, pair in enumerate(pairs, start=1):
        clean, _ = load_audio(pair.clean_path)
        noisy, _ = load_audio(pair.noisy_path)
        if clean.shape != noisy.shape:
            raise ValueError(f"Clean/noisy lengths differ for {pair.source_id}.")
        probs = gmm.predict_proba(log_mel_frames(compute_stft(noisy, config), config))
        items.append((clean, noisy, probs, pair))
        if index % 200 == 0 or index == len(pairs):
            print(f"[load] {index}/{len(pairs)}")
    return items


def crop_batch(items, indices, crop: int, rng: np.random.Generator):
    clean = np.zeros((len(indices), crop), dtype=np.float32)
    noisy = np.zeros((len(indices), crop), dtype=np.float32)
    probs = np.stack([items[i][2] for i in indices]).astype(np.float32)
    for row, i in enumerate(indices):
        c, n = items[i][0], items[i][1]
        if c.size > crop:
            start = int(rng.integers(0, c.size - crop + 1))
            c, n = c[start : start + crop], n[start : start + crop]
        clean[row, : c.size] = c
        noisy[row, : n.size] = n
    return clean, noisy, probs


def batch_tensors(torch, clean, noisy, config: STFTConfig, device):
    clean_t = torch.as_tensor(clean, device=device)
    noisy_t = torch.as_tensor(noisy, device=device)
    clean_spec = torch_stft(clean_t, config.n_fft, config.hop_length)
    noisy_spec = torch_stft(noisy_t, config.n_fft, config.hop_length)
    noise_spec = torch_stft(noisy_t - clean_t, config.n_fft, config.hop_length)
    target = clean_spec.abs() / (clean_spec.abs() + noise_spec.abs() + 1e-10)
    features = torch_normalized_log_magnitude(noisy_spec)
    return (
        features[:, None],
        target.clamp(0.0, 1.0)[:, None],
        noisy_spec.abs()[:, None],
        clean_spec.abs()[:, None],
        noisy_spec,
    )


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    error = np.sum((reference - estimate) ** 2) + 1e-12
    return float(10 * np.log10((np.sum(reference**2) + 1e-12) / error))


def validate(torch, model, items, config: STFTConfig, device) -> dict[str, float]:
    model.eval()
    losses, improvements = [], []
    with torch.no_grad():
        for clean, noisy, probs, _ in items:
            features, target, noisy_mag, clean_mag, noisy_spec = batch_tensors(
                torch, clean[None], noisy[None], config, device
            )
            cond = torch.as_tensor(probs[None], device=device)
            mask = model(features, cond)
            loss, _ = mask_loss(mask, target, noisy_mag, clean_mag)
            losses.append(float(loss))
            enhanced = torch_istft(
                noisy_spec * mask[:, 0], clean.size, config.n_fft, config.hop_length
            )[0].cpu().numpy()
            improvements.append(snr_db(clean, enhanced) - snr_db(clean, noisy))
    return {
        "val_loss": float(np.mean(losses)),
        "val_snr_improvement_db": float(np.mean(improvements)),
    }


def train(args: argparse.Namespace) -> dict:
    torch = import_torch()
    dataset_dir = args.dataset_dir.expanduser()
    run_dir = ensure_outside_dataset(dataset_dir, args.output_dir or default_run_dir(dataset_dir))
    checkpoint_path = refuse_overwrite(run_dir / "zfnet.pt", overwrite=args.overwrite)
    gmm_path = (args.gmm or run_dir / "gmm.joblib").expanduser()
    if not gmm_path.is_file():
        raise FileNotFoundError(f"GMM checkpoint not found: {gmm_path}. Run train_gmm.py first.")

    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = STFTConfig()

    gmm = NoiseGMM.load(gmm_path)
    train_items = load_split(load_pairs(dataset_dir, split="train", limit=args.limit), gmm, config)
    val_items = load_split(load_pairs(dataset_dir, split="validation", limit=args.val_limit), gmm, config)

    model_config = ZFNetConfig(
        n_classes=len(gmm.classes),
        freq_bins=config.freq_bins,
        base_channels=args.base_channels,
    )
    model = ZFNetMask(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    crop = int(round(args.crop_sec * SAMPLE_RATE))
    print(
        f"[model] {MODEL_VERSION} params={count_parameters(model):,} device={device} "
        f"train={len(train_items)} val={len(val_items)}"
    )

    train_config = {
        "created_utc": utc_now(),
        "model_version": MODEL_VERSION,
        "dataset_dir": str(dataset_dir),
        "gmm_path": str(gmm_path),
        "noise_classes": gmm.classes,
        "model_config": model_config.__dict__,
        "parameters": count_parameters(model),
        "device": device,
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    write_json(run_dir / "train_config.json", train_config)

    history: list[dict] = []
    best_loss, best_epoch, stale = np.inf, 0, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_items))
        batches = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        if args.max_steps_per_epoch:
            batches = batches[: args.max_steps_per_epoch]
        epoch_start, train_losses = time.time(), []
        for step, indices in enumerate(batches, start=1):
            clean, noisy, probs = crop_batch(train_items, indices, crop, rng)
            features, target, noisy_mag, clean_mag, _ = batch_tensors(torch, clean, noisy, config, device)
            mask = model(features, torch.as_tensor(probs, device=device))
            loss, _ = mask_loss(mask, target, noisy_mag, clean_mag)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss))
            if step % 25 == 0:
                print(f"  epoch {epoch} step {step}/{len(batches)} loss {np.mean(train_losses[-25:]):.4f}")

        metrics = validate(torch, model, val_items, config, device)
        scheduler.step(metrics["val_loss"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.time() - epoch_start, 1),
        }
        history.append(row)
        improved = metrics["val_loss"] < best_loss - 1e-5
        if improved:
            best_loss, best_epoch, stale = metrics["val_loss"], epoch, 0
            save_zfnet_checkpoint(
                checkpoint_path,
                model,
                noise_classes=gmm.classes,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                sample_rate=SAMPLE_RATE,
                model_version=MODEL_VERSION,
                extra={"epoch": epoch, **metrics},
            )
        else:
            stale += 1
        print(
            f"[epoch {epoch}] train {row['train_loss']:.4f} val {metrics['val_loss']:.4f} "
            f"val dSNR {metrics['val_snr_improvement_db']:+.2f} dB "
            f"({row['epoch_seconds']}s){' *saved' if improved else ''}"
        )
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if stale >= args.patience:
            print(f"[early stop] no val improvement for {args.patience} epochs")
            break

    summary = {
        "model_version": MODEL_VERSION,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "best_val_snr_improvement_db": next(
            (h["val_snr_improvement_db"] for h in history if h["epoch"] == best_epoch), None
        ),
        "epochs_run": len(history),
        "seconds": round(time.time() - started, 1),
        "checkpoint": str(checkpoint_path),
    }
    write_json(run_dir / "train_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = train(args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ZF-Net training failed: {exc}", file=sys.stderr)
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
