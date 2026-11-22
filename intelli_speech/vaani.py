"""Read-only loaders for the Vaani Atypical (Telugu) corpus and the typical Vaani Telugu copy.

The atypical corpus is gated on Hugging Face (ARTPARK-IISc/Vaani-Atypical-Speech-Corpus,
research use only). Log in with your own account in Colab (``huggingface_hub.login()``);
this module never handles tokens.

Audio is requested with ``decode=False`` and decoded here with soundfile, so behaviour
does not depend on the ``datasets`` version's audio backend.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

ATYPICAL_REPO = "ARTPARK-IISc/Vaani-Atypical-Speech-Corpus"
ATYPICAL_CONFIG = "Telugu"
ATYPICAL_COLUMNS = (
    "audio",
    "speaker_id",
    "language",
    "prompt_file_type",
    "prompt_file_name",
    "transcription",
    "difficulty_level",
    "verbatim_transcription",
    "duration_s",
)
TYPICAL_REQUIRED_COLUMNS = ("filename", "speakerID", "transcript")


class VaaniDependencyError(RuntimeError):
    """Raised when the Hugging Face ``datasets`` package is unavailable."""


@dataclass(frozen=True)
class AudioRecord:
    """One recording with its decoded audio and metadata (no modification of sources)."""

    record_id: str
    speaker_id: str
    waveform: np.ndarray
    sample_rate: int
    channels: int
    sha1: str
    metadata: dict[str, Any]


def decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int, int]:
    """Decode audio bytes -> (float32 samples [frames, channels], sample_rate, channels)."""
    import soundfile as sf

    audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio, int(sample_rate), int(audio.shape[1])


def pcm_sha1(audio: np.ndarray) -> str:
    """Content hash of the decoded samples (detects duplicates across file names)."""
    return hashlib.sha1(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def load_atypical_telugu(cache_dir: str | Path | None = None):
    """Load the Telugu config of the Vaani Atypical corpus with undecoded audio."""
    try:
        from datasets import Audio, load_dataset
    except Exception as exc:
        raise VaaniDependencyError("Install the Hugging Face datasets package: pip install datasets") from exc
    dataset = load_dataset(
        ATYPICAL_REPO,
        ATYPICAL_CONFIG,
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    missing = [c for c in ATYPICAL_COLUMNS if c not in dataset.column_names]
    if missing:
        raise ValueError(f"Vaani Atypical is missing expected columns: {missing}")
    return dataset.cast_column("audio", Audio(decode=False))


def iter_atypical(dataset) -> Iterator[AudioRecord]:
    """Yield decoded records one at a time (never the whole corpus in memory)."""
    for index, row in enumerate(dataset):
        audio_field = row["audio"] or {}
        metadata = {k: row[k] for k in ATYPICAL_COLUMNS if k != "audio"}
        metadata["source_path"] = audio_field.get("path")
        try:
            data = audio_field.get("bytes")
            if data is None and audio_field.get("path"):
                data = Path(audio_field["path"]).read_bytes()
            if data is None:
                raise ValueError("no audio bytes or path")
            audio, sample_rate, channels = decode_audio_bytes(data)
        except Exception as exc:  # report, do not stop the audit
            metadata["decode_error"] = f"{type(exc).__name__}: {exc}"
            audio, sample_rate, channels = np.zeros((0, 1), np.float32), 0, 0
        yield AudioRecord(
            record_id=f"atyp_te_{index:04d}",
            speaker_id=str(row["speaker_id"]),
            waveform=audio,
            sample_rate=sample_rate,
            channels=channels,
            sha1=pcm_sha1(audio),
            metadata=metadata,
        )


def load_typical_vaani_metadata(directory: str | Path) -> pd.DataFrame:
    """Read ``metadata.csv`` of the typical Vaani Telugu copy (handles a UTF-8 BOM)."""
    folder = Path(directory).expanduser()
    frame = pd.read_csv(folder / "metadata.csv", encoding="utf-8-sig", dtype=str)
    frame.columns = [str(c).strip().lstrip("﻿") for c in frame.columns]
    missing = [c for c in TYPICAL_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Typical Vaani metadata is missing columns: {missing}")
    return frame


def iter_typical_vaani(directory: str | Path, metadata: pd.DataFrame | None = None) -> Iterator[AudioRecord]:
    """Yield decoded typical Vaani recordings listed in metadata.csv."""
    folder = Path(directory).expanduser()
    frame = load_typical_vaani_metadata(folder) if metadata is None else metadata
    for _, row in frame.iterrows():
        path = folder / str(row["filename"]).strip()
        meta = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        try:
            audio, sample_rate, channels = decode_audio_bytes(path.read_bytes())
        except Exception as exc:  # report, do not stop the audit
            meta["decode_error"] = f"{type(exc).__name__}: {exc}"
            audio, sample_rate, channels = np.zeros((0, 1), np.float32), 0, 0
        yield AudioRecord(
            record_id=f"typ_vaani_{Path(str(row['filename'])).stem}",
            speaker_id=str(row["speakerID"]),
            waveform=audio,
            sample_rate=sample_rate,
            channels=channels,
            sha1=pcm_sha1(audio),
            metadata=meta,
        )
