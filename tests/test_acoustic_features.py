from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("parselmouth")

from intelli_speech.acoustic_features import (
    energy_vad,
    extract_recording_features,
    temporal_features,
)

SR = 16000


def synthetic_speech(
    *,
    syllable_rate: float = 4.0,
    syllables_per_phrase: int = 6,
    phrases: int = 5,
    pause_s: float = 0.4,
    f0: float = 150.0,
    f0_swing_st: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Harmonic syllable bursts (Hann-shaped), grouped into phrases separated by silence."""
    rng = np.random.default_rng(seed)
    period = 1.0 / syllable_rate
    syl_len = 0.6 * period
    pieces = [np.zeros(int(0.3 * SR))]
    for p in range(phrases):
        for s in range(syllables_per_phrase):
            n = int(syl_len * SR)
            t = np.arange(n) / SR
            semitone = f0_swing_st * np.sin(2 * np.pi * (p * syllables_per_phrase + s) / 7.0)
            freq = f0 * 2 ** (semitone / 12)
            tone = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 8))
            burst = tone * np.hanning(n) * 0.3
            gap = np.zeros(int((period - syl_len) * SR))
            pieces += [burst, gap]
        pieces.append(np.zeros(int(pause_s * SR)))
    x = np.concatenate(pieces)
    return (x + 1e-4 * rng.standard_normal(x.size)).astype(np.float32)


def test_vad_and_pause_statistics_recover_design() -> None:
    x = synthetic_speech(pause_s=0.5, phrases=5)
    mask = energy_vad(x, SR)
    feats = temporal_features(mask)
    # 4 internal phrase pauses of ~0.5 s (+ syllable gaps of ~0.1 s, below the 250 ms threshold)
    minutes = feats["active_duration_s"] / 60
    assert feats["pause_rate_per_min_250ms"] * minutes == pytest.approx(4, abs=1)
    assert feats["pause_median_s_250ms"] == pytest.approx(0.5, abs=0.12)
    assert 0.3 < feats["speech_activity_ratio"] < 1.0  # Hann bursts: quiet edges fall below VAD


def test_longer_pauses_increase_pause_measures() -> None:
    short = extract_recording_features(synthetic_speech(pause_s=0.3), SR, include_voice=False, include_spectral=False)
    long = extract_recording_features(synthetic_speech(pause_s=0.9), SR, include_voice=False, include_spectral=False)
    assert long["pause_median_s_250ms"] > short["pause_median_s_250ms"]
    assert long["pause_time_fraction_250ms"] > short["pause_time_fraction_250ms"]
    assert long["speech_rate_nuclei_per_s"] < short["speech_rate_nuclei_per_s"]


def test_syllable_rate_and_f0() -> None:
    feats = extract_recording_features(synthetic_speech(syllable_rate=4.0), SR, include_voice=False, include_spectral=False)
    assert feats["articulation_rate_nuclei_per_s"] == pytest.approx(4.0, rel=0.35)
    assert feats["f0_median_hz"] == pytest.approx(150, rel=0.05)
    monotone_range = feats["f0_range_p5_p95_st"]
    varied = extract_recording_features(synthetic_speech(f0_swing_st=4.0), SR, include_voice=False, include_spectral=False)
    assert varied["f0_range_p5_p95_st"] > monotone_range + 2


def test_full_feature_set_has_channel_voice_and_spectral_keys() -> None:
    feats = extract_recording_features(synthetic_speech(), SR)
    for key in ("feature_version", "effective_bandwidth_hz", "full_scale_verdict", "hnr_mean_db", "cpps_db",
                "spectral_centroid_hz_median", "ltas_slope_db_per_oct", "mfcc1_sd", "intensity_sd_db"):
        assert key in feats, key
    assert feats["hnr_mean_db"] > 5  # clean harmonic signal


def test_rejects_short_or_multichannel_input() -> None:
    with pytest.raises(ValueError):
        extract_recording_features(np.zeros(100, np.float32), SR)
    with pytest.raises(ValueError):
        extract_recording_features(np.zeros((SR, 2), np.float32), SR)
