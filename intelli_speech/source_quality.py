"""Quality screening of typical-speech source clips for synthetic data (Phase 3).

Vaani is crowd-sourced phone audio, so source clips are screened *before* any synthetic
transform; noisy clips are rejected rather than denoised (denoising would bake model
artefacts into the training targets). Per clip:

  snr_db_estimate        median level of VAD speech frames minus the median level of
                         non-speech frames (a coarse, documented estimate, not a true SNR)
  effective_bandwidth_hz 99 % cumulative-energy frequency
  full_scale_verdict     from ``audio_integrity.full_scale_analysis``
  speech_time_s / speech_activity_ratio  from the Exp 1 energy VAD

Thresholds are explicit in ``QualityConfig`` and written with every run. The filter
does not delete files; it only marks ``quality_pass`` and the first failing reasons.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import acoustic_features as af


@dataclass(frozen=True)
class QualityConfig:
    # Pilot (2026-11-22): Vaani 99 %-energy bandwidth median ~1.6 kHz; the atypical
    # speakers span 1.2-5.0 kHz on the same measure, so 1.2 kHz is the floor. SNR estimate
    # median ~15.6 dB; 12 dB rejects only the noisiest part (review on pilot quantiles).
    min_snr_db: float = 12.0
    min_bandwidth_hz: float = 1200.0
    min_speech_time_s: float = 1.5
    min_activity_ratio: float = 0.40
    reject_clipping: bool = True


def clip_quality(x: np.ndarray, sr: int) -> dict[str, Any]:
    audio = np.asarray(x, dtype=np.float32)
    db = af.frame_levels_db(audio, sr)
    mask = af.energy_vad(audio, sr)
    n = min(db.size, mask.size)
    db, mask = db[:n], mask[:n]
    speech, rest = db[mask], db[~mask]
    snr = float(np.median(speech) - np.median(rest)) if speech.size and rest.size else None
    temporal = af.temporal_features(mask)
    channel = af.channel_descriptors(audio, sr)
    return {
        "snr_db_estimate": round(snr, 2) if snr is not None else None,
        "effective_bandwidth_hz": channel["effective_bandwidth_hz"],
        "full_scale_verdict": channel["full_scale_verdict"],
        "rms_dbfs": channel["rms_dbfs"],
        "speech_time_s": temporal.get("speech_time_s", 0.0),
        "speech_activity_ratio": temporal.get("speech_activity_ratio"),
    }


def failing_reasons(q: dict[str, Any], config: QualityConfig) -> list[str]:
    reasons = []
    if q.get("snr_db_estimate") is None or q["snr_db_estimate"] < config.min_snr_db:
        reasons.append("low_snr")
    if (q.get("effective_bandwidth_hz") or 0) < config.min_bandwidth_hz:
        reasons.append("narrow_bandwidth")
    if config.reject_clipping and q.get("full_scale_verdict") == "clipping_evidence":
        reasons.append("clipping")
    if (q.get("speech_time_s") or 0) < config.min_speech_time_s:
        reasons.append("little_speech")
    if q.get("speech_activity_ratio") is None or q["speech_activity_ratio"] < config.min_activity_ratio:
        reasons.append("low_activity")
    return reasons


def assign_speaker_splits(speakers: pd.DataFrame, *, salt: str, fractions=(0.8, 0.1, 0.1)) -> pd.Series:
    """Deterministic speaker-level train/val/test split, stratified by gender.

    ``speakers`` has columns ``speakerID`` and ``gender``. Within each gender, speakers are
    ordered by salted SHA-1 and cut by the fractions (val/test get at least one speaker
    when a stratum has >= 3 speakers).
    """
    out = {}
    for _, group in speakers.groupby(speakers["gender"].fillna("unknown").astype(str).str.lower()):
        ordered = sorted(group["speakerID"].astype(str), key=lambda s: hashlib.sha1(f"{salt}|{s}".encode()).hexdigest())
        n = len(ordered)
        n_val = max(1, round(fractions[1] * n)) if n >= 3 else 0
        n_test = max(1, round(fractions[2] * n)) if n >= 3 else 0
        n_train = n - n_val - n_test
        for i, s in enumerate(ordered):
            out[s] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return speakers["speakerID"].astype(str).map(out)


def config_dict(config: QualityConfig) -> dict[str, Any]:
    return asdict(config)
