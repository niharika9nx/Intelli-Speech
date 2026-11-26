"""DSP temporal module: typical → synthetic-atypical timing (Phase 3), version 0.2.

Implements the Exp 1 class-1 pattern (slower articulation, more / longer / more variable
pauses, lower speech-activity ratio) on a typical source clip:

1. Energy VAD (same as Exp 1) marks speech, internal pauses and edge silence (10 ms frames).
2. **One continuous, time-varying WSOLA pass** over the whole clip (no per-run joins).
   The local stretch factor is 1 in pauses/edges; inside speech it is larger in steady
   (vowel-like, low spectral flux) frames and smaller in transients (onsets, bursts), so
   the result sounds slow rather than slow-motion. The mean speech factor equals
   ``stretch``. Window 30 ms, search ±15 ms (covers pitch periods down to ~67 Hz).
3. Pause plan: target count = ``pause_rate_per_min`` × output active minutes (solved
   iteratively). Existing pauses are lengthened to max(original, draw); extra pauses are
   inserted only at real dips inside speech (≥ ``DIP_DB`` below the run's median level
   for ≥ ``DIP_MIN_S``), away from run edges. Durations: log-normal (``pause_mu``,
   ``pause_sigma``) truncated to [0.25 s, ``max_pause_s``].
4. Added pause audio is the clip's own background noise, built with overlapping
   equal-power crossfades (continuous, no pumping), never digital silence.
5. Every splice uses an equal-power crossfade (``XFADE_S``).
6. Frame labels (10 ms) for ZF-Net localisation: 0 edge, 1 speech (stretched),
   2 original pause, 3 lengthened part of a pause, 4 inserted pause.

Deterministic for a given ``seed``. Pitch is untouched (speaker-specific prosody is a
separate module). v0.1 (per-run stretching, butt joins, gated noise loops, uniform
stretch) produced audible clicks and slow-motion quality and is superseded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from . import acoustic_features as af

TRANSFORM_VERSION = "temporal-0.2"
HOP_S = af.FRAME_HOP_S
LABELS = {"edge": 0, "speech": 1, "pause": 2, "pause_lengthened": 3, "pause_inserted": 4}
MIN_PAUSE_S = 0.25
INSERT_EDGE_GUARD_S = 0.25
XFADE_S = 0.010
WIN_S = 0.030
TOL_S = 0.015
TRANSIENT_WEIGHT = 0.35   # transients stretch this fraction as much as steady frames
CONTOUR_SMOOTH_S = 0.05
DIP_DB = 10.0
DIP_MIN_S = 0.03
MAX_STRETCH = 2.5


@dataclass(frozen=True)
class TemporalParams:
    stretch: float = 1.0              # >= 1: mean speech lengthening factor (1/articulation-rate ratio)
    pause_rate_per_min: float = 0.0   # target internal pauses >= 0.25 s per minute of active output
    pause_mu: float = float(np.log(0.5))  # log-normal location of pause duration (s)
    pause_sigma: float = 0.4
    max_pause_s: float = 3.0
    seed: int = 0


# --------------------------------------------------------------------------- time stretch


def wsola(x: np.ndarray, sr: int, factor, win_s: float = WIN_S, tol_s: float = TOL_S,
          frame_hop_s: float = HOP_S, return_map: bool = False):
    """Deterministic, pitch-preserving, optionally time-varying WSOLA.

    ``factor`` is a scalar or an array with one value per ``frame_hop_s`` input frame
    (>1 lengthens). Output length ~= sum(factor) * frame_hop_s * sr. With
    ``return_map`` also returns (output sample of each synthesis hop, input sample it
    was taken from), for label mapping.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(round(win_s * sr)) // 2 * 2
    hs, tol = n // 2, int(round(tol_s * sr))
    fh = int(round(frame_hop_s * sr))
    if np.isscalar(factor):
        contour = np.full(max(1, int(np.ceil(x.size / fh))), float(factor))
    else:
        contour = np.asarray(factor, dtype=np.float64)
    contour = np.clip(contour, 0.25, 4.0)

    positions = [0.0]
    while positions[-1] < x.size:
        f = contour[min(int(positions[-1] // fh), contour.size - 1)]
        positions.append(positions[-1] + hs / f)
    frames = len(positions)
    xp = np.pad(x, (tol, n + 2 * tol))
    window = np.hanning(n + 2)[1:-1]
    y = np.zeros(frames * hs + n)
    wsum = np.zeros_like(y)
    delta = 0
    taken = np.zeros(frames)
    for k in range(frames):
        start = int(round(positions[k])) + tol + delta
        start = min(max(start, 0), xp.size - n)
        y[k * hs: k * hs + n] += xp[start: start + n] * window
        wsum[k * hs: k * hs + n] += window
        taken[k] = start - tol
        if k + 1 < frames:
            natural = xp[start + hs: start + hs + n]
            nominal = int(round(positions[k + 1])) + tol
            lo = max(nominal - tol, 0)
            region = xp[lo: nominal + tol + n]
            corr = np.correlate(region, natural, mode="valid") if natural.size == n and region.size >= n else np.zeros(1)
            delta = (int(np.argmax(corr)) + lo - nominal) if np.any(natural) and corr.size > 1 else 0
    # positions[-1] is the first analysis point at/after the end of x: output ends there
    over = positions[-1] - x.size
    last_f = contour[min(int(positions[-2] // fh), contour.size - 1)] if frames > 1 else 1.0
    out_len = int(round((frames - 1) * hs - over * last_f))
    # never extend past the point whose source reaches the end of x (no padded zeros)
    src_centre = np.maximum.accumulate(taken) + n / 2
    end_out = float(np.interp(x.size, src_centre, np.arange(frames) * hs + n / 2))
    out_len = max(1, min(out_len, int(round(end_out)), (frames - 1) * hs + n))
    y = (y / np.maximum(wsum, 1e-3))[:out_len].astype(np.float32)
    if return_map:
        return y, np.arange(frames) * hs, np.clip(taken, 0, x.size - 1)
    return y


def stretch_contour(x: np.ndarray, sr: int, mask: np.ndarray, stretch: float) -> np.ndarray:
    """Per-10 ms factor: 1 outside speech; steady speech frames stretch more than transients.

    Mean factor over speech frames == ``stretch`` (duration-exact on average).
    """
    contour = np.ones(mask.size)
    stretch = float(np.clip(stretch, 1.0, MAX_STRETCH))
    if stretch <= 1.0001 or not mask.any():
        return contour
    hop = int(HOP_S * sr)
    n = mask.size
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (0, 2 * hop)), 2 * hop)[::hop][:n]
    spec = np.abs(np.fft.rfft(frames * np.hanning(2 * hop), axis=1))
    logspec = np.log(spec + 1e-6)
    flux = np.r_[0.0, np.sqrt(np.mean(np.diff(logspec, axis=0) ** 2, axis=1))]
    level = af.frame_levels_db(x, sr)[:n]
    if level.size < n:
        level = np.pad(level, (0, n - level.size), mode="edge")
    sp = mask.astype(bool)
    f_sp, l_sp = flux[sp], level[sp]
    flux_n = (f_sp - np.percentile(f_sp, 10)) / max(np.percentile(f_sp, 90) - np.percentile(f_sp, 10), 1e-6)
    loud = (l_sp - np.percentile(l_sp, 10)) / max(np.percentile(l_sp, 90) - np.percentile(l_sp, 10), 1e-6)
    steady = np.clip(0.6 * (1 - np.clip(flux_n, 0, 1)) + 0.4 * np.clip(loud, 0, 1), 0, 1)
    weight = TRANSIENT_WEIGHT + (1 - TRANSIENT_WEIGHT) * steady
    k = max(1, int(CONTOUR_SMOOTH_S / HOP_S))
    weight_full = np.zeros(n)
    weight_full[sp] = weight
    smoothed = np.convolve(weight_full, np.ones(k) / k, mode="same")
    norm = np.convolve(sp.astype(float), np.ones(k) / k, mode="same")
    w = np.where(sp, smoothed / np.maximum(norm, 1e-6), 0.0)
    scale = (stretch - 1.0) / max(w[sp].mean(), 1e-6)
    contour[sp] = 1.0 + scale * w[sp]
    return contour


# --------------------------------------------------------------------------- splicing


def _xfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Equal-power crossfade of the last n samples of a with the first n of b."""
    n = min(n, a.size, b.size)
    if n < 2:
        return np.concatenate([a, b])
    t = np.linspace(0.0, np.pi / 2, n)
    mid = a[-n:] * np.cos(t) + b[:n] * np.sin(t)
    return np.concatenate([a[:-n], mid.astype(np.float32), b[n:]])


class NoiseBed:
    """The clip's own background noise, continuous and click-free.

    Pieces are cut only from inside single contiguous non-speech regions (minus a
    ``MARGIN_S`` guard next to speech, so no speech tails leak in) and joined with
    overlapping equal-power crossfades. Joining all non-speech samples end to end would
    create waveform jumps at every region boundary (the v0.2-pre glitch).
    """

    MARGIN_S = 0.03
    PIECE_S = 0.35
    XF_S = 0.03

    def __init__(self, x: np.ndarray, mask_samples: np.ndarray, sr: int, rng: np.random.Generator):
        self.sr, self.rng = sr, rng
        self.xf = int(self.XF_S * sr)
        margin = int(self.MARGIN_S * sr)
        regions, i, n = [], 0, mask_samples.size
        while i < n:
            j = i
            while j < n and mask_samples[j] == mask_samples[i]:
                j += 1
            if not mask_samples[i]:
                a, b = i + (margin if i > 0 else 0), j - (margin if j < n else 0)
                if b - a >= 3 * self.xf:
                    regions.append(x[a:b].astype(np.float32))
            i = j
        if sum(r.size for r in regions) < int(0.3 * sr):  # too little clean silence: floor-level noise
            floor_db = float(np.percentile(af.frame_levels_db(x, sr), 10))
            regions = [(10 ** (floor_db / 20) * rng.standard_normal(int(0.6 * sr))).astype(np.float32)]
            self.synthetic = True
        else:
            self.synthetic = False
        self.regions = regions
        self.weights = np.array([r.size for r in regions], dtype=float) / sum(r.size for r in regions)

    def _piece(self) -> np.ndarray:
        region = self.regions[int(self.rng.choice(len(self.regions), p=self.weights))]
        size = min(region.size, max(3 * self.xf, int(self.PIECE_S * self.sr)))
        start = int(self.rng.integers(0, region.size - size + 1))
        return region[start:start + size]

    def take(self, n: int) -> np.ndarray:
        out = self._piece().copy()
        while out.size < n + self.xf:
            out = _xfade(out, self._piece(), self.xf)
        return out[:n]


# --------------------------------------------------------------------------- pause plan


def _draw_pause(params: TemporalParams, rng: np.random.Generator) -> float:
    for _ in range(100):
        d = float(np.exp(rng.normal(params.pause_mu, params.pause_sigma)))
        if MIN_PAUSE_S <= d <= params.max_pause_s:
            return d
    return float(np.clip(np.exp(params.pause_mu), MIN_PAUSE_S, params.max_pause_s))


def _runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    out, i = [], 0
    while i < mask.size:
        j = i
        while j < mask.size and mask[j] == mask[i]:
            j += 1
        out.append((bool(mask[i]), i, j))
        i = j
    return out


def _dip_frames(level: np.ndarray, runs: list[tuple[bool, int, int]]) -> list[int]:
    """Input frames at the centre of real dips inside speech runs (candidate pause sites)."""
    guard = int(INSERT_EDGE_GUARD_S / HOP_S)
    min_len = max(1, int(round(DIP_MIN_S / HOP_S)))
    out: list[int] = []
    for is_sp, a, b in runs:
        if not is_sp or b - a <= 2 * guard + min_len:
            continue
        seg = level[a:b]
        low = seg < np.median(seg) - DIP_DB
        i = guard
        while i < seg.size - guard:
            if low[i]:
                j = i
                while j < seg.size - guard and low[j]:
                    j += 1
                if j - i >= min_len:
                    out.append(a + (i + j) // 2)
                i = j
            else:
                i += 1
    return out


# --------------------------------------------------------------------------- transform


def transform(x: np.ndarray, sr: int, params: TemporalParams) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (audio, frame_labels, info). Deterministic for a given ``params.seed``."""
    rng = np.random.default_rng(params.seed)
    audio = np.asarray(x, dtype=np.float32)
    hop = int(HOP_S * sr)
    mask = af.energy_vad(audio, sr)
    runs = _runs(mask)
    speech_runs = [r for r in runs if r[0]]
    if not speech_runs:
        raise ValueError("No speech detected; cannot apply temporal transform.")
    first, last = speech_runs[0][1], speech_runs[-1][2]
    mask_samples = np.repeat(mask, hop)[: audio.size]
    mask_samples = np.pad(mask_samples, (0, audio.size - mask_samples.size))
    bed = NoiseBed(audio, mask_samples.astype(bool), sr, rng)

    # input frame classes: 0 edge, 1 speech, 2 internal pause
    cls = np.where(mask, LABELS["speech"], LABELS["pause"]).astype(np.int8)
    cls[:first] = LABELS["edge"]
    cls[last:] = LABELS["edge"]

    # 1) one continuous time-varying stretch
    contour = stretch_contour(audio, sr, mask, params.stretch)
    y, out_pos, in_pos = wsola(audio, sr, contour, return_map=True)
    n_out = int(np.ceil(y.size / hop))
    in_frame_of_out = np.interp(np.arange(n_out) * hop, out_pos, in_pos) // hop
    labels = cls[np.clip(in_frame_of_out.astype(int), 0, cls.size - 1)].astype(np.int8)

    def out_frame(in_frame: float) -> int:
        return int(np.clip(np.interp(in_frame * hop, in_pos, out_pos) // hop, 0, n_out - 1))

    # 2) pause plan
    speech_out = float((labels == LABELS["speech"]).sum() * HOP_S)
    internal = [(a, b) for is_sp, a, b in runs if not is_sp and first <= a and b <= last]
    targets = [max((b - a) * HOP_S, _draw_pause(params, rng)) for a, b in internal]
    inserts: list[float] = []
    for _ in range(4):  # fixed point: count depends on active duration, which depends on pauses
        active = speech_out + sum(targets) + sum(inserts)
        need = max(0, int(round(params.pause_rate_per_min * active / 60.0)) - sum(t >= MIN_PAUSE_S for t in targets))
        while len(inserts) < need:
            inserts.append(_draw_pause(params, rng))
        inserts = inserts[:need]

    level = af.frame_levels_db(audio, sr)
    if level.size < mask.size:
        level = np.pad(level, (0, mask.size - level.size), mode="edge")
    dips = _dip_frames(level[: mask.size], runs)
    chosen = sorted(dips[i] for i in (rng.permutation(len(dips))[: len(inserts)] if dips else []))

    # 3) edits in output frames: (frame, seconds, label)
    edits: list[tuple[int, float, int]] = []
    for (a, b), t in zip(internal, targets):
        extra = t - (b - a) * HOP_S
        if extra > 0.02:
            edits.append((out_frame((a + b) / 2), extra, LABELS["pause_lengthened"]))
    for frame, t in zip(chosen, inserts):
        edits.append((out_frame(frame), t, LABELS["pause_inserted"]))

    # 4) apply edits right-to-left with equal-power crossfades
    xf = int(XFADE_S * sr)
    for frame, seconds, label in sorted(edits, key=lambda e: e[0], reverse=True):
        pos = min(frame * hop, y.size)
        n_ins = int(round(seconds * sr))
        noise = bed.take(n_ins + 2 * xf)
        y = _xfade(_xfade(y[:pos], noise, xf), y[pos:], xf)
        labels = np.concatenate([labels[:frame], np.full(int(round(n_ins / hop)), label, np.int8), labels[frame:]])

    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    n_frames = int(np.ceil(y.size / hop))
    labels = np.pad(labels, (0, max(0, n_frames - labels.size)), mode="edge")[:n_frames]
    info = {
        "version": TRANSFORM_VERSION,
        "params": asdict(params),
        "input_s": round(audio.size / sr, 3), "output_s": round(y.size / sr, 3),
        "speech_in_s": round(float(mask[first:last].sum() * HOP_S), 3),
        "speech_out_s": round(speech_out, 3),
        "speech_contour_mean": round(float(contour[mask].mean()), 4) if mask.any() else 1.0,
        "speech_contour_range": [round(float(contour[mask].min()), 3), round(float(contour[mask].max()), 3)] if mask.any() else [1, 1],
        "existing_pauses": len(internal),
        "inserted_pauses": len(chosen), "inserted_requested": len(inserts),
        "insertion_shortfall": len(inserts) - len(chosen), "dip_candidates": len(dips),
        "splices": len(edits),
        "noise_bed_synthetic": bed.synthetic,
        "label_seconds": {name: round(float((labels == v).sum() * HOP_S), 3) for name, v in LABELS.items()},
    }
    return y, labels, info


def click_score(y: np.ndarray, sr: int) -> float:
    """Coarse splice-click proxy: 99.9th percentile of 1 ms high-frequency energy relative
    to its running 50 ms median (median ignores the click itself). Compare output vs input;
    a hard waveform discontinuity raises it sharply, smooth crossfades do not.
    """
    d = np.diff(np.asarray(y, dtype=np.float64))
    m = int(0.001 * sr)
    if d.size < 60 * m:
        return 0.0
    e = (d[: d.size // m * m] ** 2).reshape(-1, m).mean(axis=1)
    k = 51
    padded = np.pad(e, (k // 2, k // 2), mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(padded, k), axis=1)
    floor = 0.01 * np.percentile(e, 95) + 1e-12  # tied to speech level, not to the share of pauses
    return float(np.percentile(e / (med + floor), 99.9))
