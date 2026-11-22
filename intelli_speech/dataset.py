from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .audio_io import resolve_metadata_audio_path

VALID_SPLITS = ("train", "validation", "test")
REQUIRED_METADATA_COLUMNS = (
    "source_id",
    "split",
    "clean_file",
    "noisy_file",
    "target_snr_db",
)
DEFAULT_METADATA_FILENAME = "metadata.csv"


@dataclass(frozen=True)
class AudioPair:
    """One clean/noisy metadata row with resolved audio paths."""

    source_id: str
    split: str
    clean_path: Path
    noisy_path: Path
    target_snr_db: float
    sample_rate: int = 16000
    duration_sec: float | None = None
    noise_environment: str | None = None
    noise_file: str | None = None
    gender: str | None = None
    speaker_id: str | None = None
    transcript: str | None = None
    row_index: int = 0


def metadata_path(
    dataset_dir: str | Path,
    *,
    filename: str = DEFAULT_METADATA_FILENAME,
) -> Path:
    """Return the metadata path for a dataset directory after existence checks."""
    path = Path(dataset_dir).expanduser() / filename
    if not path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {path}")
    return path


def load_metadata(
    dataset_dir: str | Path,
    *,
    filename: str = DEFAULT_METADATA_FILENAME,
) -> pd.DataFrame:
    """Load and validate paired-dataset metadata without modifying the CSV."""
    dataset_path = Path(dataset_dir).expanduser()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    path = metadata_path(dataset_path, filename=filename)
    try:
        metadata = pd.read_csv(
            path,
            dtype={
                "source_id": str,
                "split": str,
                "clean_file": str,
                "noisy_file": str,
            },
        )
    except Exception as exc:
        raise ValueError(f"Could not read dataset metadata {path}: {exc}") from exc

    missing = set(REQUIRED_METADATA_COLUMNS) - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing)}"
        )
    if metadata.empty:
        raise ValueError(f"Metadata contains no rows: {path}")

    metadata = metadata.copy()
    metadata["source_id"] = metadata["source_id"].astype(str).str.strip()
    metadata["split"] = metadata["split"].astype(str).str.strip()
    metadata["clean_file"] = metadata["clean_file"].astype(str).str.strip()
    metadata["noisy_file"] = metadata["noisy_file"].astype(str).str.strip()

    for column in ("source_id", "clean_file", "noisy_file"):
        if metadata[column].eq("").any() or metadata[column].eq("nan").any():
            raise ValueError(f"Metadata column {column!r} contains empty values.")
    if metadata[["clean_file", "noisy_file"]].isna().any().any():
        raise ValueError("Metadata audio paths contain missing values.")
    if not metadata["noisy_file"].is_unique:
        raise ValueError("Metadata noisy_file values must be unique.")

    invalid_splits = sorted(set(metadata["split"]) - set(VALID_SPLITS))
    if invalid_splits:
        raise ValueError(
            f"Metadata contains invalid splits: {invalid_splits}; "
            f"expected one of {VALID_SPLITS}."
        )

    try:
        metadata["target_snr_db"] = pd.to_numeric(
            metadata["target_snr_db"], errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Metadata target_snr_db must be numeric.") from exc
    if not np.isfinite(metadata["target_snr_db"].to_numpy(dtype=float)).all():
        raise ValueError("Metadata target_snr_db contains non-finite values.")

    if "sample_rate" in metadata.columns:
        metadata["sample_rate"] = pd.to_numeric(
            metadata["sample_rate"], errors="raise"
        )
        if (metadata["sample_rate"] <= 0).any():
            raise ValueError("Metadata sample_rate values must be positive.")

    return metadata


def filter_split(metadata: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return one validated split as a reset-index copy."""
    normalized = str(split).strip()
    if normalized not in VALID_SPLITS:
        raise ValueError(
            f"Unknown split {split!r}; expected one of {VALID_SPLITS}."
        )
    filtered = metadata.loc[metadata["split"] == normalized].copy()
    if filtered.empty:
        raise ValueError(f"Split {normalized!r} contains no metadata rows.")
    return filtered.reset_index(drop=True)


def split_counts(metadata: pd.DataFrame) -> dict[str, int]:
    """Count metadata rows per split in a stable order."""
    counts = metadata["split"].value_counts().to_dict()
    return {split: int(counts[split]) for split in VALID_SPLITS if split in counts}


def iter_pairs(
    dataset_dir: str | Path,
    *,
    metadata: pd.DataFrame | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> Iterator[AudioPair]:
    """Yield resolved clean/noisy pairs for a dataset or preloaded metadata."""
    dataset_path = Path(dataset_dir).expanduser()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    frame = load_metadata(dataset_path) if metadata is None else metadata.copy()
    if split is not None:
        frame = filter_split(frame, split)
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be a positive integer.")
        frame = frame.head(int(limit))

    for row_index, row in frame.iterrows():
        clean_path = resolve_metadata_audio_path(dataset_path, row["clean_file"])
        noisy_path = resolve_metadata_audio_path(dataset_path, row["noisy_file"])
        yield AudioPair(
            source_id=str(row["source_id"]),
            split=str(row["split"]),
            clean_path=clean_path,
            noisy_path=noisy_path,
            target_snr_db=float(row["target_snr_db"]),
            sample_rate=_optional_int(row.get("sample_rate"), default=16000),
            duration_sec=_optional_float(row.get("duration_sec")),
            noise_environment=_optional_string(row.get("noise_environment")),
            noise_file=_optional_string(row.get("noise_file")),
            gender=_optional_string(row.get("gender")),
            speaker_id=_optional_string(row.get("speaker_id")),
            transcript=_optional_string(row.get("transcript")),
            row_index=int(row_index),
        )


def load_pairs(
    dataset_dir: str | Path,
    *,
    metadata: pd.DataFrame | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> list[AudioPair]:
    """Materialize :func:`iter_pairs` for callers that need random access."""
    return list(
        iter_pairs(
            dataset_dir,
            metadata=metadata,
            split=split,
            limit=limit,
        )
    )


def _optional_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: object, *, default: int) -> int:
    if value is None or pd.isna(value):
        return default
    return int(value)


__all__ = [
    "AudioPair",
    "DEFAULT_METADATA_FILENAME",
    "REQUIRED_METADATA_COLUMNS",
    "VALID_SPLITS",
    "filter_split",
    "iter_pairs",
    "load_metadata",
    "load_pairs",
    "metadata_path",
    "split_counts",
]
