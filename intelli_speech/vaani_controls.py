"""Bounded, deterministic, speaker-balanced typical-Telugu control set from regular Vaani.

Source: Hugging Face ``ARTPARK-IISc/Vaani`` (gated, CC-BY-4.0). The Telugu config is
~2,326 h, so nothing is bulk-downloaded:

1. ``scan_metadata`` reads only non-audio parquet columns from a deterministic, evenly
   spread subset of Telugu shards (column projection = small range reads).
2. ``select_controls`` picks speakers and clips by salted SHA-1 rank (reproducible,
   independent of shard order), with optional gender interleaving and per-speaker caps.
3. ``fetch_selected`` reads the audio column only for row groups that contain selected
   rows, decodes, converts to 16 kHz mono PCM16 and records provenance.

The filesystem is injectable (``huggingface_hub.HfFileSystem`` in Colab, a local
fsspec filesystem in tests).
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

VAANI_REPO = "ARTPARK-IISc/Vaani"
TELUGU_GLOBS = (
    f"datasets/{VAANI_REPO}/audio/Telugu/*.parquet",
    f"datasets/{VAANI_REPO}/audio/Telugu/**/*.parquet",
    f"datasets/{VAANI_REPO}/**/Telugu/**/*.parquet",
)
METADATA_COLUMNS = (
    "language",
    "duration",
    "speakerID",
    "gender",
    "state",
    "district",
    "pincode",
    "isTranscriptionAvailable",
    "transcript",
    "referenceImage",
    "UtteranceSequenceID",
)
TARGET_SAMPLE_RATE = 16000
SALT = "intelli-speech-typical-controls-v2"


@dataclass(frozen=True)
class SelectionConfig:
    n_speakers: int = 40
    max_per_speaker: int = 10
    max_total: int = 400
    min_per_speaker: int = 3
    min_duration_s: float = 2.0
    max_duration_s: float = 30.0
    require_transcript: bool = True
    balance_gender: bool = True
    salt: str = SALT


def rank_key(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def list_telugu_shards(fs, globs: Iterable[str] = TELUGU_GLOBS) -> list[str]:
    for pattern in globs:
        found = sorted(set(fs.glob(pattern)))
        if found:
            return found
    return []


def spread_subset(items: list[str], k: int | None) -> list[str]:
    """Deterministic, evenly spaced subset (keeps district/order diversity of sorted shards)."""
    if k is None or k >= len(items):
        return list(items)
    index = np.unique(np.linspace(0, len(items) - 1, k).round().astype(int))
    return [items[i] for i in index]


def _truthy(value: Any) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in ("true", "1", "yes", "y", "t")


def scan_metadata(fs, shards: list[str], columns: Iterable[str] = METADATA_COLUMNS) -> pd.DataFrame:
    """Read non-audio columns of the given shards, keeping (shard, row_group, row) positions."""
    import pyarrow.parquet as pq

    frames = []
    for shard in shards:
        with fs.open(shard, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            available = [c for c in columns if c in parquet.schema_arrow.names]
            for rg in range(parquet.num_row_groups):
                table = parquet.read_row_group(rg, columns=available).to_pandas()
                table["_shard"] = shard
                table["_row_group"] = rg
                table["_row"] = np.arange(len(table))
                frames.append(table)
    if not frames:
        return pd.DataFrame(columns=[*columns, "_shard", "_row_group", "_row"])
    return pd.concat(frames, ignore_index=True)


def select_controls(metadata: pd.DataFrame, config: SelectionConfig = SelectionConfig()) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deterministic speaker-balanced selection. Never relaxes the caps to hit targets."""
    frame = metadata.copy()
    report: dict[str, Any] = {"scanned_rows": int(len(frame))}
    if "language" in frame:
        frame = frame[frame["language"].astype(str).str.strip().str.lower() == "telugu"]
    report["telugu_rows"] = int(len(frame))
    duration = pd.to_numeric(frame.get("duration"), errors="coerce")
    frame = frame[(duration >= config.min_duration_s) & (duration <= config.max_duration_s)]
    report["duration_ok_rows"] = int(len(frame))
    if config.require_transcript:
        has_text = frame.get("transcript", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str).str.strip() != ""
        flag = frame.get("isTranscriptionAvailable")
        if flag is not None:
            has_text &= flag.map(_truthy) | flag.isna()
        frame = frame[has_text]
    report["transcribed_rows"] = int(len(frame))
    frame = frame[frame["speakerID"].notna() & (frame["speakerID"].astype(str).str.strip() != "")]
    frame = frame.assign(speakerID=frame["speakerID"].astype(str).str.strip())

    # One row per (speaker, utterance id) to prevent duplicates across shards.
    utt = frame["UtteranceSequenceID"].astype(str) if "UtteranceSequenceID" in frame else frame["_row"].astype(str)
    frame = frame.assign(_utt=utt).drop_duplicates(subset=["speakerID", "_utt"])
    counts = frame.groupby("speakerID").size()
    eligible = counts[counts >= config.min_per_speaker].index.tolist()
    report["speakers_seen"] = int(counts.size)
    report["eligible_speakers"] = len(eligible)

    ranked = sorted(eligible, key=lambda s: rank_key(config.salt, s))
    if config.balance_gender and "gender" in frame:
        gender_of = frame.groupby("speakerID")["gender"].agg(lambda g: str(g.mode().iloc[0]).strip().lower() if g.notna().any() else "unknown")
        queues: dict[str, list[str]] = {}
        for s in ranked:
            queues.setdefault(gender_of.get(s, "unknown"), []).append(s)
        order, keys = [], sorted(queues)
        while any(queues.values()):
            for key in keys:
                if queues[key]:
                    order.append(queues[key].pop(0))
        ranked = order

    chosen_rows, total, speakers = [], 0, []
    for speaker in ranked:
        if len(speakers) >= config.n_speakers or total >= config.max_total:
            break
        rows = frame[frame["speakerID"] == speaker].copy()
        rows["_rank"] = [rank_key(config.salt, speaker, u) for u in rows["_utt"]]
        take = rows.sort_values("_rank").head(min(config.max_per_speaker, config.max_total - total))
        if len(take) < config.min_per_speaker:
            break
        chosen_rows.append(take)
        speakers.append(speaker)
        total += len(take)

    selected = pd.concat(chosen_rows, ignore_index=True) if chosen_rows else frame.head(0)
    selected["selection_rank"] = np.arange(len(selected))
    report.update(
        {
            "selected_speakers": len(speakers),
            "selected_rows": int(len(selected)),
            "target_speakers": config.n_speakers,
            "shortfall": len(speakers) < config.n_speakers,
            "config": config.__dict__,
        }
    )
    if "gender" in selected:
        report["selected_gender_speakers"] = selected.drop_duplicates("speakerID")["gender"].astype(str).value_counts().to_dict()
    if "district" in selected:
        report["selected_districts"] = int(selected["district"].nunique())
    return selected, report


def estimate_audio_bytes(fs, selected: pd.DataFrame, audio_column: str = "audio") -> int:
    """Compressed size of the audio column chunks that must be read for the selection."""
    import pyarrow.parquet as pq

    total = 0
    for shard, groups in selected.groupby("_shard")["_row_group"]:
        with fs.open(shard, "rb") as handle:
            meta = pq.ParquetFile(handle).metadata
            names = [meta.schema.column(i).path.split(".")[0] for i in range(meta.num_columns)]
            for rg in sorted(set(groups)):
                row_group = meta.row_group(int(rg))
                total += sum(row_group.column(i).total_compressed_size for i, n in enumerate(names) if n == audio_column)
    return int(total)


def to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    if sample_rate != TARGET_SAMPLE_RATE:
        g = gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        mono = resample_poly(mono, TARGET_SAMPLE_RATE // g, int(sample_rate) // g)
    return np.clip(mono, -1.0, 1.0).astype(np.float32)


def fetch_selected(fs, selected: pd.DataFrame, output_dir: Path, *, source_revision: str | None, audio_column: str = "audio") -> pd.DataFrame:
    """Download only the needed audio, write 16 kHz mono PCM16 WAVs, return the manifest."""
    import pyarrow.parquet as pq
    import soundfile as sf

    audio_dir = Path(output_dir) / "audio"
    records, seen_hashes = [], {}
    for (shard, rg), rows in selected.groupby(["_shard", "_row_group"], sort=True):
        with fs.open(shard, "rb") as handle:
            column = pq.ParquetFile(handle).read_row_group(int(float(rg)), columns=[audio_column]).column(audio_column).to_pylist()
        for _, row in rows.iterrows():
            cell = column[int(float(row["_row"]))] or {}
            data = cell.get("bytes") if isinstance(cell, dict) else None
            entry = {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in row.items() if not str(k).startswith("_rank")}
            entry.update({"source_repo": VAANI_REPO, "source_revision": source_revision, "source_shard": shard,
                          "source_row_group": int(float(rg)), "source_row": int(float(row["_row"])), "license": "CC-BY-4.0"})
            try:
                audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
                mono = to_mono_16k(audio, rate)
                digest = hashlib.sha1(mono.tobytes()).hexdigest()
                if digest in seen_hashes:
                    entry.update({"status": "duplicate_skipped", "duplicate_of": seen_hashes[digest]})
                    records.append(entry)
                    continue
                control_id = f"ctl_{len(seen_hashes):04d}"
                speaker_dir = audio_dir / str(row["speakerID"])
                speaker_dir.mkdir(parents=True, exist_ok=True)
                path = speaker_dir / f"{control_id}.wav"
                sf.write(str(path), mono, TARGET_SAMPLE_RATE, subtype="PCM_16")
                seen_hashes[digest] = control_id
                entry.update({
                    "status": "ok", "control_id": control_id, "file": str(path.relative_to(output_dir)),
                    "original_sample_rate": int(rate), "original_channels": int(audio.shape[1]),
                    "resampled": int(rate) != TARGET_SAMPLE_RATE, "sample_rate": TARGET_SAMPLE_RATE,
                    "duration_s": round(mono.size / TARGET_SAMPLE_RATE, 4), "pcm_sha1": digest,
                })
            except Exception as exc:
                entry.update({"status": "decode_error", "error": f"{type(exc).__name__}: {exc}"})
            records.append(entry)
    return pd.DataFrame(records)
