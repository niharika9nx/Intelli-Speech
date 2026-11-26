"""Experiment 2 (Stage A) measures: articulation, formant dynamics, rhythm, pause windows.

Implements docs/research/EXP2_PLAN.md v0.2.1 §5–6 and §13.4. All thresholds below are
pre-registered there; do not change them after seeing results.

Measurement only — no synthetic articulation transforms live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import acoustic_features as af

FEATURE_VERSION = "exp2-features-0.2"
SR = 16000
HOP_S = 0.010

# §5 channel conditioning
ANALYSIS_BAND_HZ = (100.0, 3400.0)
CUTOFF_GATE_HZ = 3400.0
CUTOFF_DB_BELOW_MAX = 40.0
# §6.1 formant tracking
CEILING_HZ = 3700.0
N_FORMANTS = 3
WINDOW_S = 0.025
F1_RANGE = (200.0, 1000.0)
F2_RANGE = (600.0, 3300.0)
B1_MAX, B2_MAX = 400.0, 600.0
NUCLEUS_WINDOW_S = 0.030
NASAL_LOW_MINUS_MID_DB = 15.0
NASAL_B1_MIN = 300.0
MIN_FRAMES_RECORDING = 20          # per-recording minimum for a cloud measure
MIN_CLIPS_SPEAKER = 3              # speaker-level sufficiency (§6.1)
MIN_FRAMES_SPEAKER = 300
# §6.3 dynamics
RUN_MIN_S = 0.060
DCT_COEFFS = 4
MIN_EXTENT_ST = 1.0
MIN_TRANSITION_S = 0.020
MIN_PAIRS = 3
# §6.4 EMS
EMS_RATE = 80
EMS_BANDS_HZ = (250.0, 500.0, 1000.0, 2000.0)   # 250 Hz is reported with a flag
EMS_MIN_S = 2.0
# §6.5 / §13.4 pause windows
WINDOW_LEN_S = 5.8
WINDOW_MIN_S = 4.0
PAUSE_MIN_S = 0.100
LONG_PAUSE_S = 1.0

PRIMARY_MEASURES = ("aavs_log", "vsa_hull_log", "f2_excursion_st_median", "f2_speed_fit_oct_s", "pause_long_share_1s_win")


# --------------------------------------------------------------------------- channel


def channel_descriptors(x: np.ndarray, sr: int = SR) -> dict[str, Any]:
    """`upper_cutoff_hz` (highest frequency within 40 dB of the PSD max) and `low_ratio_db`."""
    from scipy.signal import welch

    x = np.asarray(x, dtype=np.float64)
    if x.size < 1024 or not np.any(x):
        return {"upper_cutoff_hz": None, "low_ratio_db": None}
    freqs, psd = welch(x, fs=sr, nperseg=1024)
    smooth_bins = max(1, int(round(200.0 / (freqs[1] - freqs[0]))))
    psd_s = np.convolve(psd, np.ones(smooth_bins) / smooth_bins, mode="same")
    db = 10 * np.log10(psd_s + 1e-20)
    above = np.flatnonzero(db >= db.max() - CUTOFF_DB_BELOW_MAX)
    low = psd[(freqs > 0) & (freqs < 300)].sum()
    mid = psd[(freqs >= 300) & (freqs < 1000)].sum()
    return {
        "upper_cutoff_hz": round(float(freqs[above[-1]]), 1),
        "low_ratio_db": round(float(10 * np.log10((low + 1e-20) / (mid + 1e-20))), 3),
    }


def harmonise(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Band-limit to the common analysis band (100–3400 Hz) for formant analysis."""
    from scipy.signal import butter, sosfiltfilt

    if sr != SR:
        raise ValueError("harmonise expects 16 kHz audio")
    x = np.asarray(x, dtype=np.float64)
    if x.size < 64:
        return x.astype(np.float32)
    sos = butter(6, ANALYSIS_BAND_HZ, btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


# --------------------------------------------------------------------------- tracking


@dataclass
class Tracks:
    times: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    voiced: np.ndarray
    intensity_db: np.ndarray
    nasal: np.ndarray
    f0_median: float | None = None


def track(x: np.ndarray, sr: int = SR, *, ceiling_hz: float = CEILING_HZ, raw: np.ndarray | None = None) -> Tracks:
    """Formants on the harmonised signal; voicing from Exp 1's two-pass pitch on `raw` (or x)."""
    parselmouth = af._parselmouth()
    snd = parselmouth.Sound(np.asarray(x, dtype=np.float64), sampling_frequency=sr)
    formant = snd.to_formant_burg(time_step=HOP_S, max_number_of_formants=N_FORMANTS,
                                  maximum_formant=ceiling_hz, window_length=WINDOW_S, pre_emphasis_from=50.0)
    times = np.asarray(formant.xs())
    get = formant.get_value_at_time
    getb = formant.get_bandwidth_at_time
    f1 = np.array([get(1, t) for t in times], dtype=np.float64)
    f2 = np.array([get(2, t) for t in times], dtype=np.float64)
    b1 = np.array([getb(1, t) for t in times], dtype=np.float64)
    b2 = np.array([getb(2, t) for t in times], dtype=np.float64)

    src = parselmouth.Sound(np.asarray(raw if raw is not None else x, dtype=np.float64), sampling_frequency=sr)
    pitch = af._pitch(src, parselmouth)
    f0, tp = pitch.selected_array["frequency"], np.asarray(pitch.xs())
    voiced = np.zeros(times.size, bool)
    if tp.size:
        idx = np.clip(np.searchsorted(tp, times), 0, tp.size - 1)
        voiced = f0[idx] > 0
    inten = src.to_intensity(minimum_pitch=75.0, time_step=HOP_S)
    i_db = np.interp(times, np.asarray(inten.xs()), inten.values[0]) if inten.values.size else np.full(times.size, -np.inf)

    # nasal-murmur heuristic on the harmonised signal (§6.1)
    nasal = np.zeros(times.size, bool)
    n = int(WINDOW_S * sr)
    xx = np.asarray(x, dtype=np.float64)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    lo_band, mid_band = (freqs >= 100) & (freqs < 500), (freqs >= 500) & (freqs < 3400)
    win = np.hanning(n)
    for k, t in enumerate(times):
        a = int(round(t * sr)) - n // 2
        if a < 0 or a + n > xx.size:
            continue
        spec = np.abs(np.fft.rfft(xx[a:a + n] * win)) ** 2
        lo, mid = spec[lo_band].sum() + 1e-20, spec[mid_band].sum() + 1e-20
        nasal[k] = (10 * np.log10(lo / mid) > NASAL_LOW_MINUS_MID_DB) and (np.isfinite(b1[k]) and b1[k] > NASAL_B1_MIN)
    f0v = f0[f0 > 0] if tp.size else np.zeros(0)
    return Tracks(times, f1, f2, b1, b2, voiced, i_db, nasal, float(np.median(f0v)) if f0v.size >= 10 else None)


def basic_valid(tr: Tracks) -> np.ndarray:
    """Range + bandwidth + voicing + not nasal (no nucleus restriction)."""
    ok = tr.voiced & ~tr.nasal
    ok &= np.isfinite(tr.f1) & np.isfinite(tr.f2) & np.isfinite(tr.b1) & np.isfinite(tr.b2)
    ok &= (tr.f1 >= F1_RANGE[0]) & (tr.f1 <= F1_RANGE[1]) & (tr.f2 >= F2_RANGE[0]) & (tr.f2 <= F2_RANGE[1])
    ok &= (tr.b1 < B1_MAX) & (tr.b2 < B2_MAX)
    return ok


def nuclei_times(raw: np.ndarray, sr: int, tr: Tracks) -> np.ndarray:
    """Exp 1 acoustic syllable nuclei, computed on the raw clip."""
    parselmouth = af._parselmouth()
    snd = parselmouth.Sound(np.asarray(raw, dtype=np.float64), sampling_frequency=sr)
    inten = snd.to_intensity(minimum_pitch=75.0, time_step=HOP_S)
    t_int, i_db = np.asarray(inten.xs()), inten.values[0]

    def voiced_at(t: float) -> bool:
        k = int(np.argmin(np.abs(tr.times - t))) if tr.times.size else 0
        return bool(tr.times.size and tr.voiced[k])

    return af.syllable_nuclei(i_db, t_int, voiced_at)


def select_frames(tr: Tracks, nuclei: np.ndarray, speech_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """§6.1 frame selection. Returns boolean selection + diagnostics."""
    diag: dict[str, Any] = {"n_frames": int(tr.times.size)}
    if tr.times.size == 0:
        return np.zeros(0, bool), {**diag, "n_selected": 0}
    near = np.zeros(tr.times.size, bool)
    for t in nuclei:
        near |= np.abs(tr.times - t) <= NUCLEUS_WINDOW_S
    sp_idx = np.clip((tr.times / HOP_S).astype(int), 0, max(speech_mask.size - 1, 0))
    in_speech = speech_mask[sp_idx] if speech_mask.size else np.zeros(tr.times.size, bool)
    finite_i = np.isfinite(tr.intensity_db)
    speech_i = tr.intensity_db[in_speech & finite_i]
    loud = finite_i & (tr.intensity_db >= (np.median(speech_i) if speech_i.size else np.inf))
    pre = tr.voiced & near & loud
    diag.update({
        "n_voiced": int(tr.voiced.sum()), "n_near_nuclei": int((tr.voiced & near).sum()), "n_candidates": int(pre.sum()),
        "share_f1_below": _share(pre, tr.f1 < F1_RANGE[0]), "share_f1_above": _share(pre, tr.f1 > F1_RANGE[1]),
        "share_f2_below": _share(pre, tr.f2 < F2_RANGE[0]), "share_f2_above": _share(pre, tr.f2 > F2_RANGE[1]),
        "share_nasal_dropped": _share(pre, tr.nasal),
    })
    sel = pre & basic_valid(tr)
    diag["n_selected"] = int(sel.sum())
    return sel, diag


def _share(base: np.ndarray, cond: np.ndarray) -> float | None:
    n = int(base.sum())
    return round(float((base & np.nan_to_num(cond, nan=False).astype(bool)).sum()) / n, 4) if n else None


# --------------------------------------------------------------------------- vowel space


def aavs_log(lf1: np.ndarray, lf2: np.ndarray) -> float | None:
    """sqrt(det cov(ln F1, ln F2)); scale-invariant under uniform formant scaling."""
    if lf1.size < MIN_FRAMES_RECORDING:
        return None
    cov = np.cov(np.vstack([lf1, lf2]))
    det = float(np.linalg.det(cov))
    return float(np.sqrt(det)) if det > 0 else 0.0


def vsa_hull_log(lf1: np.ndarray, lf2: np.ndarray, keep: float = 0.90) -> float | None:
    """Convex-hull area of the central `keep` fraction (Mahalanobis-trimmed) of the log cloud."""
    from scipy.spatial import ConvexHull, QhullError

    if lf1.size < MIN_FRAMES_RECORDING:
        return None
    pts = np.column_stack([lf1, lf2])
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T)
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.0
    d = np.einsum("ij,jk,ik->i", pts - mu, inv, pts - mu)
    kept = pts[d <= np.quantile(d, keep)]
    if kept.shape[0] < 3:
        return 0.0
    try:
        return float(ConvexHull(kept).volume)
    except (QhullError, ValueError):
        return 0.0


def diag_extent_log(lf1: np.ndarray, lf2: np.ndarray) -> dict[str, float | None]:
    if lf1.size < MIN_FRAMES_RECORDING:
        return {"diag_front_back_log": None, "diag_open_close_log": None}
    a, b = lf2 - lf1, lf1 + lf2
    return {"diag_front_back_log": float(np.percentile(a, 95) - np.percentile(a, 5)),
            "diag_open_close_log": float(np.percentile(b, 95) - np.percentile(b, 5))}


# --------------------------------------------------------------------------- dynamics


def dct_fit(y: np.ndarray, n_coef: int = DCT_COEFFS) -> np.ndarray:
    """Least-squares fit of a cosine (DCT-II) basis with `n_coef` terms."""
    n = y.size
    k = np.arange(min(n_coef, n))
    basis = np.cos(np.pi * np.outer(np.arange(n) + 0.5, k) / n)
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return basis @ coef


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < mask.size:
        if mask[i]:
            j = i
            while j < mask.size and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def formant_dynamics(tr: Tracks, nuclei: np.ndarray) -> dict[str, Any]:
    """§6.3: per voiced run, DCT-fit ln F2; excursion and speed between consecutive nuclei."""
    valid = basic_valid(tr)
    excursions, speeds, stiff = [], [], []
    min_len = int(round(RUN_MIN_S / HOP_S))
    for a, b in _runs(valid):
        if b - a < min_len:
            continue
        t = tr.times[a:b]
        fit = dct_fit(np.log(tr.f2[a:b]))
        inside = nuclei[(nuclei >= t[0]) & (nuclei <= t[-1])]
        for n1, n2 in zip(inside[:-1], inside[1:]):
            seg = (t >= n1) & (t <= n2)
            if seg.sum() < 2:
                continue
            st = 12 * (fit[seg] - fit[seg].min()) / np.log(2)
            extent = float(st.max())
            excursions.append(extent)
            dt = float(n2 - n1)
            if extent >= MIN_EXTENT_ST and dt >= MIN_TRANSITION_S:
                # §6.3: median absolute slope of the fitted ln F2 within the transition (oct/s)
                slope = float(np.median(np.abs(np.diff(fit[seg])) / HOP_S / np.log(2)))
                speeds.append(slope)
                stiff.append(slope / extent)
    ok = len(excursions) >= MIN_PAIRS
    return {
        "n_transition_pairs": len(excursions),
        "n_speed_transitions": len(speeds),
        "f2_excursion_st_median": float(np.median(excursions)) if ok else None,
        "f2_speed_fit_oct_s": float(np.median(speeds)) if len(speeds) >= MIN_PAIRS else None,
        "f2_stiffness_exploratory": float(np.median(stiff)) if len(stiff) >= MIN_PAIRS else None,
    }


# --------------------------------------------------------------------------- EMS (C1)


def ems(x: np.ndarray, sr: int = SR, band_hz: float | None = None) -> dict[str, float | None]:
    """Envelope modulation spectrum: peak frequency and 3–5 Hz / 0.5–10 Hz energy ratio."""
    from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

    x = np.asarray(x, dtype=np.float64)
    if x.size < EMS_MIN_S * sr:
        return {"peak_hz": None, "ratio_3_5": None}
    if band_hz is not None:
        lo, hi = band_hz / np.sqrt(2), min(band_hz * np.sqrt(2), ANALYSIS_BAND_HZ[1])
        x = sosfiltfilt(butter(4, (lo, hi), btype="bandpass", fs=sr, output="sos"), x)
    env = np.abs(hilbert(x))
    env = sosfiltfilt(butter(4, 10.0, btype="lowpass", fs=sr, output="sos"), env)
    env = resample_poly(env, EMS_RATE, sr)
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(env.size))) ** 2
    freqs = np.fft.rfftfreq(env.size, 1.0 / EMS_RATE)
    band = (freqs >= 0.5) & (freqs <= 10.0)
    total = spec[band].sum()
    if total <= 0:
        return {"peak_hz": None, "ratio_3_5": None}
    mid = (freqs >= 3.0) & (freqs <= 5.0)
    return {"peak_hz": float(freqs[band][np.argmax(spec[band])]), "ratio_3_5": float(spec[mid].sum() / total)}


def speech_only(x: np.ndarray, mask: np.ndarray, sr: int = SR) -> np.ndarray:
    hop = int(HOP_S * sr)
    keep = np.repeat(mask, hop)[: x.size]
    keep = np.pad(keep, (0, max(0, x.size - keep.size)))
    return np.asarray(x)[keep.astype(bool)]


# --------------------------------------------------------------------------- pause windows (H4)


def pause_windows(mask: np.ndarray, duration_s: float) -> dict[str, Any]:
    """§13.4 windows; counts only pauses fully inside a window (≥ PAUSE_MIN_S)."""
    starts: list[float] = []
    if duration_s >= WINDOW_LEN_S:
        s = 0.0
        while s + WINDOW_LEN_S <= duration_s + 1e-9:
            starts.append(s)
            s += WINDOW_LEN_S
        if duration_s - s >= WINDOW_MIN_S:
            starts.append(s)
    elif duration_s >= WINDOW_MIN_S:
        starts.append(0.0)
    # internal pauses (between first and last speech frame)
    speech = np.flatnonzero(mask)
    pauses: list[tuple[float, float]] = []
    if speech.size:
        inner = mask[speech[0]: speech[-1] + 1]
        for a, b in _runs(~inner):
            pauses.append(((speech[0] + a) * HOP_S, (speech[0] + b) * HOP_S))
    n = n_long = 0
    for s in starts:
        e = min(s + WINDOW_LEN_S, duration_s)
        for a, b in pauses:
            if a >= s and b <= e and (b - a) >= PAUSE_MIN_S:
                n += 1
                n_long += int((b - a) > LONG_PAUSE_S)
    return {"n_windows": len(starts), "pause_count_win": n, "pause_long_count_win": n_long,
            "pause_long_share_1s_win": (n_long / n) if n else None}


# --------------------------------------------------------------------------- per recording


def extract(x: np.ndarray, sr: int = SR, *, ceiling_hz: float = CEILING_HZ, cutoff_gate_hz: float = CUTOFF_GATE_HZ,
            return_tracks: bool = False) -> dict[str, Any] | tuple[dict[str, Any], Tracks | None, np.ndarray, np.ndarray]:
    """All Stage A measures for one mono 16 kHz recording.

    Formant measures are None when the clip fails the cutoff gate (`gate_pass` False) or
    lacks frames; timing (H4) and EMS are always computed on the raw clip.
    """
    raw = np.asarray(x, dtype=np.float32)
    if raw.ndim != 1 or raw.size < int(0.5 * sr):
        raise ValueError("Expected a mono recording of at least 0.5 s.")
    out: dict[str, Any] = {"feature_version": FEATURE_VERSION, "duration_s": round(raw.size / sr, 3),
                           "ceiling_hz": ceiling_hz, "cutoff_gate_hz": cutoff_gate_hz}
    out.update(channel_descriptors(raw, sr))
    if np.any(raw):
        out.update(af.channel_descriptors(raw, sr))
    else:  # Exp 1 helper fails on all-zero input; Exp 1 code is left unchanged
        out.update({"rms_dbfs": None, "effective_bandwidth_hz": None, "near_full_scale_fraction": None, "full_scale_verdict": None})
    mask = af.energy_vad(raw, sr)
    out.update(pause_windows(mask, raw.size / sr))
    for key, sig in (("full", raw), ("speech_only", speech_only(raw, mask, sr))):
        e = ems(harmonise(sig, sr) if sig.size >= 64 else sig, sr)
        out[f"ems_{key}_peak_hz"], out[f"ems_{key}_ratio_3_5"] = e["peak_hz"], e["ratio_3_5"]
    hb = harmonise(raw, sr)
    for band in EMS_BANDS_HZ:
        e = ems(hb, sr, band)
        out[f"ems_b{int(band)}_ratio_3_5"] = e["ratio_3_5"]
    cutoff = out.get("upper_cutoff_hz")
    out["gate_pass"] = bool(cutoff is not None and cutoff >= cutoff_gate_hz)
    tr: Tracks | None = None
    sel = np.zeros(0, bool)
    nuclei = np.zeros(0)
    try:
        tr = track(hb, sr, ceiling_hz=ceiling_hz, raw=raw)
        nuclei = nuclei_times(raw, sr, tr)
    except Exception as exc:  # Praat failures are recorded, not fatal
        out["formant_error"] = f"{type(exc).__name__}: {exc}"
    active_s = float(mask.sum()) * HOP_S
    out["nuclei_per_s"] = round(nuclei.size / active_s, 4) if active_s else None
    if tr is not None:
        out["f0_median_hz"] = tr.f0_median
        sel, diag = select_frames(tr, nuclei, mask)
        out.update({f"sel_{k}": v for k, v in diag.items()})
        out["selected_frames_per_s"] = round(int(sel.sum()) / active_s, 4) if active_s else None
    out.setdefault("f0_median_hz", None)
    for key in ("aavs_log", "vsa_hull_log", "diag_front_back_log", "diag_open_close_log", "f2_excursion_st_median",
                "f2_speed_fit_oct_s", "f2_stiffness_exploratory", "n_transition_pairs", "n_speed_transitions"):
        out.setdefault(key, None)
    if tr is not None and out["gate_pass"]:
        lf1, lf2 = np.log(tr.f1[sel]), np.log(tr.f2[sel])
        out["aavs_log"] = aavs_log(lf1, lf2)
        out["vsa_hull_log"] = vsa_hull_log(lf1, lf2)
        out.update(diag_extent_log(lf1, lf2))
        out.update(formant_dynamics(tr, nuclei))
    if return_tracks:
        return out, tr, sel, nuclei
    return out
