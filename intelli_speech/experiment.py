"""Experiment output-directory helpers shared by the training/evaluation scripts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATASET_DIR = Path("/content/drive/MyDrive/Intelli-Speech-data/paired_dataset_v3")
DEFAULT_RUN_NAME = "zfnet_gmm_v1"


def default_run_dir(dataset_dir: Path, run_name: str = DEFAULT_RUN_NAME) -> Path:
    return Path(dataset_dir).expanduser().parent / "experiments" / run_name


def ensure_outside_dataset(dataset_dir: Path, output_dir: Path) -> Path:
    """Resolve ``output_dir`` and refuse any location inside the dataset directory."""
    dataset_path = Path(dataset_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination == dataset_path or dataset_path in destination.parents:
        raise ValueError("Output directory must be outside the dataset directory.")
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"Output path is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def refuse_overwrite(path: Path, *, overwrite: bool) -> Path:
    """Refuse to replace an existing artifact unless ``overwrite`` is set."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
