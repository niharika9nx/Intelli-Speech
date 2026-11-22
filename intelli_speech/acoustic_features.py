"""Experiment 1 acoustic features (per recording), temporal/prosodic first.

Every feature is a *measurement*, not an impairment marker. Channel descriptors are
returned alongside so analyses can control for recording conditions.

Families
--------
temporal  : speech-activity ratio, internal pause rate/duration (150/250/500 ms),
            speech-run statistics, acoustic syllable-nucleus speech & articulation rate,
            inter-nucleus interval variability (CV, nPVI). No transcripts are used.
prosodic  : F0 (two-pass speaker-adaptive range) median, SD and P5–P95 range in
            semitones, voiced fraction, octave-jump rate; intensity SD; syllabic
            energy-modulation peak (2–8 Hz); nucleus-prominence variability.
voice     : HNR (cross-correlation), CPPS (Praat), jitter/shimmer (exploratory).
spectral  : band-limited (80–4000 Hz) centroid, 85 % roll-off, flux, LTAS slope,
            MFCC 1–13 SDs after per-recording mean removal.
channel   : effective bandwidth, rms dBFS, near-full-scale fraction.

Methods are standard but simplified; parameters are explicit module constants.
The syllable-nucleus detector follows the idea of de Jong & Wempe (2009): intensity
peaks that are sufficiently prominent and voiced.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FEATURE_VERSION = "exp1-features-0.1"
FRAME_HOP_S = 0.010
FRAME_WIN_S = 0.025
VAD_MAX_DROP_DB = 35.0        # frames this far below the 95th-percentile level are non-speech
VAD_FLOOR_MARGIN_DB = 6.0     # ...or within this margin of the 10th-percentile floor
MIN_SPEECH_S = 0.06
MIN_PAUSE_S = 0.10
PAUSE_THRESHOLDS_S = (0.15, 0.25, 0.50)
NUCLEUS_MIN_DIP_DB = 2.0
NUCLEUS_DROP_FROM_MAX_DB = 25.0
SPECTRAL_BAND_HZ = (80.0, 4000.0)
ARTICULATION_PAUSE_S = 0.25


class AcousticDependencyError(RuntimeError):
    pass


def _parselmouth():
    try:
        import parselmouth
    except Exception as exc:
        raise AcousticDependencyError("pip install praat-parselmouth") from exc
    return parselmouth


# --------------------------------------------------------------------------- framing / VAD


def frame_levels_db(x: np.ndarray, sr: int, win_s: float = FRAME_WIN_S, hop_s: float = FRAME_HOP_S) -> np.ndarray:
    win, hop = int(win_s * sr), int(hop_s * sr)
    if x.size < win:
        return np.array([10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)])
    n = 1 + (x.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx].astype(np.float64)
    return 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)


def _fill_short(mask: np.ndarray, value: bool, min_len: int) -> np.ndarray:
    """Flip runs of ``value`` shorter than ``min_len`` frames (internal runs only)."""
    out = mask.copy()
    n = out.size
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        if out[i] == value and (j - i) < min_len and i > 0 and j < n:
            out[i:j] = not value
        i = j
    return out


def energy_vad(x: np.ndarray, sr: int) -> np.ndarray:
    """Adaptive energy VAD → boolean speech mask per 10 ms frame (documented heuristic)."""
    db = frame_levels_db(x, sr)
    p95, p10 = np.percentile(db, 95), np.percentile(db, 10)
    threshold = max(p95 - VAD_MAX_DROP_DB, p10 + VAD_FLOOR_MARGIN_DB)
    mask = db > threshold
    mask = _fill_short(mask, False, int(MIN_PAUSE_S / FRAME_HOP_S))
    mask = _fill_short(mask, True, int(MIN_SPEECH_S / FRAME_HOP_S))
    return mask


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    runs, n, i = [], mask.size, 0
    while i < n:
        j = i
        while j < n and mask[j] == mask[i]:
            j += 1
        if mask[i] == value:
            runs.append((i, j))
        i = j
    return runs


def temporal_features(mask: np.ndarray, hop_s: float = FRAME_HOP_S) -> dict[str, Any]:
    speech = _runs(mask, True)
    if not speech:
        return {"speech_frames": 0}
    first, last = speech[0][0], speech[-1][1]
    active_dur = (last - first) * hop_s  # leading/trailing silence excluded
    speech_time = float(mask[first:last].sum()) * hop_s
    internal_pauses = [(e - s) * hop_s for s, e in _runs(mask[first:last], False)]
    run_durs = np.array([(e - s) * hop_s for s, e in speech])
    out: dict[str, Any] = {
        "active_duration_s": round(active_dur, 3),
        "speech_time_s": round(speech_time, 3),
        "speech_activity_ratio": round(speech_time / active_dur, 4) if active_dur else None,
        "speech_run_mean_s": round(float(run_durs.mean()), 4),
        "speech_run_cv": round(float(run_durs.std() / run_durs.mean()), 4) if run_durs.mean() else None,
    }
    minutes = active_dur / 60 if active_dur else None
    for th in PAUSE_THRESHOLDS_S:
        p = np.array([d for d in internal_pauses if d >= th])
        tag = f"{int(th * 1000)}ms"
        out[f"pause_rate_per_min_{tag}"] = round(p.size / minutes, 3) if minutes else None
        out[f"pause_median_s_{tag}"] = round(float(np.median(p)), 4) if p.size else None
        out[f"pause_p90_s_{tag}"] = round(float(np.percentile(p, 90)), 4) if p.size else None
        out[f"pause_iqr_s_{tag}"] = round(float(np.subtract(*np.percentile(p, [75, 25]))), 4) if p.size >= 2 else None
        out[f"pause_time_fraction_{tag}"] = round(float(p.sum()) / active_dur, 4) if active_dur else None
    return out


# --------------------------------------------------------------------------- prosody / nuclei


def _pitch(snd, parselmouth):
    """Two-pass speaker-adaptive F0 (first pass 60–600 Hz; second q15*0.75 .. q85*1.5)."""
    first = snd.to_pitch_ac(time_step=FRAME_HOP_S, pitch_floor=60.0, pitch_ceiling=600.0)
    f0 = first.selected_array["frequency"]
    voiced = f0[f0 > 0]
    if voiced.size < 10:
        return first
    floor = max(40.0, 0.75 * np.percentile(voiced, 15))
    ceiling = min(1000.0, 1.5 * np.percentile(voiced, 85))
    return snd.to_pitch_ac(time_step=FRAME_HOP_S, pitch_floor=floor, pitch_ceiling=ceiling)


def syllable_nuclei(intensity_db: np.ndarray, times: np.ndarray, voiced_at) -> np.ndarray:
    """Times of prominent, voiced intensity peaks (de Jong & Wempe-style heuristic)."""
    if intensity_db.size < 3:
        return np.zeros(0)
    threshold = max(np.median(intensity_db), intensity_db.max() - NUCLEUS_DROP_FROM_MAX_DB)
    peaks = [i for i in range(1, intensity_db.size - 1)
             if intensity_db[i] >= intensity_db[i - 1] and intensity_db[i] > intensity_db[i + 1] and intensity_db[i] > threshold]
    kept: list[int] = []
    for p in peaks:
        if kept:
            dip = intensity_db[kept[-1]: p + 1].min()
            if min(intensity_db[kept[-1]], intensity_db[p]) - dip < NUCLEUS_MIN_DIP_DB:
                if intensity_db[p] > intensity_db[kept[-1]]:
                    kept[-1] = p
                continue
        kept.append(p)
    return np.array([times[i] for i in kept if voiced_at(times[i])])


def prosodic_features(x: np.ndarray, sr: int, mask: np.ndarray) -> dict[str, Any]:
    parselmouth = _parselmouth()
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    pitch = _pitch(snd, parselmouth)
    f0 = pitch.selected_array["frequency"]
    t_pitch = pitch.xs()
    voiced = f0 > 0
    out: dict[str, Any] = {}

    speech_time = float(mask.sum()) * FRAME_HOP_S
    runs_all = _runs(mask, True)
    if runs_all:
        first, last = runs_all[0][0], runs_all[-1][1]
        long_pauses = sum((e - s) for s, e in _runs(mask[first:last], False) if (e - s) * FRAME_HOP_S >= ARTICULATION_PAUSE_S)
        phonation_time = ((last - first) - long_pauses) * FRAME_HOP_S
    else:
        phonation_time = 0.0
    if voiced.sum() >= 10:
        st = 12 * np.log2(f0[voiced] / np.median(f0[voiced]))
        out.update({
            "f0_median_hz": round(float(np.median(f0[voiced])), 2),
            "f0_sd_st": round(float(np.std(st)), 4),
            "f0_range_p5_p95_st": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 4),
            "voiced_fraction_of_speech": round(float(voiced.sum()) * pitch.time_step / speech_time, 4) if speech_time else None,
        })
        d = np.abs(np.diff(12 * np.log2(np.where(voiced, f0, np.nan))))
        consecutive = np.isfinite(d)
        out["f0_jump_rate_gt6st"] = round(float((d[consecutive] > 6).mean()), 5) if consecutive.any() else None

    intensity = snd.to_intensity(minimum_pitch=75.0, time_step=FRAME_HOP_S)
    i_db = intensity.values[0]
    t_int = intensity.xs()
    speech_idx = np.clip(((t_int - FRAME_WIN_S / 2) / FRAME_HOP_S).astype(int), 0, mask.size - 1)
    in_speech = mask[speech_idx]
    if in_speech.sum() >= 10:
        out["intensity_sd_db"] = round(float(np.std(i_db[in_speech])), 4)
        env = i_db - i_db.mean()
        spec = np.abs(np.fft.rfft(env * np.hanning(env.size)))
        freqs = np.fft.rfftfreq(env.size, FRAME_HOP_S)
        band = (freqs >= 2.0) & (freqs <= 8.0)
        if band.any():
            out["energy_modulation_peak_hz"] = round(float(freqs[band][np.argmax(spec[band])]), 3)

    def voiced_at(t: float) -> bool:
        k = int(np.argmin(np.abs(t_pitch - t))) if t_pitch.size else 0
        return bool(t_pitch.size and voiced[k])

    nuclei = syllable_nuclei(i_db, t_int, voiced_at)
    speech_runs = _runs(mask, True)
    if speech_runs:
        active = (speech_runs[-1][1] - speech_runs[0][0]) * FRAME_HOP_S
        out["nuclei_count"] = int(nuclei.size)
        out["speech_rate_nuclei_per_s"] = round(nuclei.size / active, 4) if active else None
        # phonation time = active time minus pauses >= ARTICULATION_PAUSE_S (de Jong & Wempe style)
        out["articulation_rate_nuclei_per_s"] = round(nuclei.size / phonation_time, 4) if phonation_time else None
        out["phonation_time_s"] = round(phonation_time, 3)
    if nuclei.size >= 4:
        # intervals only within speech runs (pauses excluded)
        run_id = np.full(mask.size, -1)
        for k, (s, e) in enumerate(speech_runs):
            run_id[s:e] = k
        ids = run_id[np.clip((nuclei / FRAME_HOP_S).astype(int), 0, mask.size - 1)]
        intervals = np.array([b - a for a, b, ia, ib in zip(nuclei[:-1], nuclei[1:], ids[:-1], ids[1:]) if ia == ib and ia >= 0])
        if intervals.size >= 3:
            out["internucleus_interval_cv"] = round(float(intervals.std() / intervals.mean()), 4)
            out["internucleus_npvi"] = round(float(100 * np.mean(np.abs(np.diff(intervals)) / ((intervals[1:] + intervals[:-1]) / 2))), 3)
        peak_db = np.interp(nuclei, t_int, i_db)
        out["nucleus_prominence_sd_db"] = round(float(np.std(peak_db)), 4)
    return out


def voice_features(x: np.ndarray, sr: int) -> dict[str, Any]:
    parselmouth = _parselmouth()
    from parselmouth.praat import call

    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    out: dict[str, Any] = {}
    harmonicity = snd.to_harmonicity_cc(time_step=FRAME_HOP_S, minimum_pitch=75.0)
    values = harmonicity.values[0]
    values = values[values > -100]
    out["hnr_mean_db"] = round(float(values.mean()), 4) if values.size else None
    try:
        # 10 ms frames + straight-line trend with "Robust" fit: ~100x faster than the
        # 2 ms / "Robust slow" default and within ~0.3 dB on test signals (documented choice).
        cepstrogram = call(snd, "To PowerCepstrogram", 60.0, 0.01, 5000.0, 50.0)
        out["cpps_db"] = round(float(call(cepstrogram, "Get CPPS", "yes", 0.02, 0.0005, 60.0, 330.0, 0.05,
                                          "Parabolic", 0.001, 0.0, "Straight", "Robust")), 4)
    except Exception:
        out["cpps_db"] = None
    try:
        points = call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)
        out["jitter_local_exploratory"] = round(float(call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)), 6)
        out["shimmer_local_exploratory"] = round(float(call([snd, points], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)), 6)
    except Exception:
        out["jitter_local_exploratory"] = out["shimmer_local_exploratory"] = None
    return out


# --------------------------------------------------------------------------- spectral / channel


def spectral_features(x: np.ndarray, sr: int, mask: np.ndarray) -> dict[str, Any]:
    from .stft import STFTConfig, compute_stft

    config = STFTConfig(n_fft=512, hop_length=int(FRAME_HOP_S * sr), sample_rate=sr)
    magnitude = np.abs(compute_stft(x.astype(np.float32), config))
    freqs = np.fft.rfftfreq(config.n_fft, 1.0 / sr)
    band = (freqs >= SPECTRAL_BAND_HZ[0]) & (freqs <= SPECTRAL_BAND_HZ[1])
    frames = min(magnitude.shape[1], mask.size)
    speech = mask[:frames]
    if speech.sum() < 10:
        return {}
    mag = magnitude[band][:, :frames][:, speech]
    f = freqs[band]
    power = mag**2 + 1e-12
    centroid = (f[:, None] * power).sum(0) / power.sum(0)
    cumulative = np.cumsum(power, axis=0) / power.sum(0)
    rolloff = f[np.argmax(cumulative >= 0.85, axis=0)]
    normed = mag / (np.linalg.norm(mag, axis=0, keepdims=True) + 1e-12)
    flux = np.sqrt((np.diff(normed, axis=1) ** 2).sum(0))
    ltas = 10 * np.log10(power.mean(1))
    octaves = np.log2(f)
    slope = np.polyfit(octaves, ltas, 1)[0]
    out = {
        "spectral_centroid_hz_median": round(float(np.median(centroid)), 2),
        "spectral_rolloff85_hz_median": round(float(np.median(rolloff)), 2),
        "spectral_flux_median": round(float(np.median(flux)), 5) if flux.size else None,
        "ltas_slope_db_per_oct": round(float(slope), 4),
    }
    try:
        import librosa

        mel = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=magnitude[:, :frames] ** 2, sr=sr, n_mels=40, fmin=SPECTRAL_BAND_HZ[0], fmax=SPECTRAL_BAND_HZ[1])), n_mfcc=13)
        mfcc = mel[:, speech]
        mfcc = mfcc - mfcc.mean(axis=1, keepdims=True)  # per-recording mean removal (channel)
        for k in range(13):
            out[f"mfcc{k + 1}_sd"] = round(float(mfcc[k].std()), 4)
    except Exception:
        pass
    return out


def channel_descriptors(x: np.ndarray, sr: int) -> dict[str, Any]:
    from .audio_integrity import full_scale_analysis

    fs = full_scale_analysis(x.astype(np.float32), is_pcm16=False)
    spectrum = np.abs(np.fft.rfft(x.astype(np.float64) * np.hanning(x.size))) ** 2 if x.size else np.zeros(1)
    cumulative = np.cumsum(spectrum) / max(spectrum.sum(), 1e-12)
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr) if x.size else np.zeros(1)
    return {
        "rms_dbfs": round(float(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)), 3),
        "effective_bandwidth_hz": round(float(freqs[int(np.searchsorted(cumulative, 0.99))]), 1),
        "near_full_scale_fraction": fs["near_full_scale_fraction"],
        "full_scale_verdict": fs["verdict"],
    }


def extract_recording_features(x: np.ndarray, sr: int, *, include_voice: bool = True, include_spectral: bool = True) -> dict[str, Any]:
    """All Experiment 1 features for one mono recording (float in [-1, 1])."""
    audio = np.asarray(x, dtype=np.float32)
    if audio.ndim != 1 or audio.size < int(0.5 * sr):
        raise ValueError("Expected a mono recording of at least 0.5 s.")
    mask = energy_vad(audio, sr)
    out: dict[str, Any] = {"feature_version": FEATURE_VERSION, "duration_s": round(audio.size / sr, 3)}
    out.update(channel_descriptors(audio, sr))
    out.update(temporal_features(mask))
    out.update(prosodic_features(audio, sr, mask))
    if include_voice:
        out.update(voice_features(audio, sr))
    if include_spectral:
        out.update(spectral_features(audio, sr, mask))
    return out
