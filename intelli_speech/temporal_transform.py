"""DSP temporal module: typical → synthetic-atypical timing (Phase 3).

Implements the Exp 1 class-1 pattern (slower articulation, more / longer / more variable
pauses, lower speech-activity ratio) on a typical source clip:

1. Energy VAD (same as Exp 1) splits the clip into edge silence, speech runs and internal pauses.
2. Each speech run is lengthened by ``stretch`` with deterministic WSOLA (pitch
   preserved). Praat PSOLA is available for listening comparisons only (not reproducible).
3. Pause plan: target count = ``pause_rate_per_min`` × output active minutes (solved
   iteratively). Existing pauses are lengthened to max(original, draw); extra pauses
   are inserted at low-energy points inside stretched speech runs. Durations are drawn
   from a log-normal (``pause_mu``, ``pause_sigma``) truncated to [0.25 s, ``max_pause_s``].
4. Added pause audio is the clip's own background noise (looped from its non-speech
   frames), never digital silence, so a model cannot detect pauses from zeros.
5. Frame labels (10 ms) mark what was changed, for ZF-Net localisation targets:
   0 edge silence, 1 speech (stretched), 2 original pause, 3 lengthened part of a pause,
   4 inserted pause.

Pitch is untouched here (pitch changes are speaker-specific → separate prosody module).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from . import acoustic_features as af

HOP_S = af.FRAME_HOP_S
LABELS = {"edge": 0, "speech": 1, "pause": 2, "pause_lengthened": 3, "pause_inserted": 4}
MIN_STRETCH_RUN_S = 0.08
FADE_S = 0.008
MIN_PAUSE_S = 0.25
INSERT_EDGE_GUARD_S = 0.25


@dataclass(frozen=True)
class TemporalParams:
    stretch: float = 1.0              # >= 1: speech-run lengthening factor (1/articulation-rate ratio)
    pause_rate_per_min: float = 0.0   # target internal pauses >= 0.25 s per minute of active output
    pause_mu: float = float(np.log(0.5))  # log-normal location of pause duration (s)
    pause_sigma: float = 0.4
    max_pause_s: float = 3.0
    seed: int = 0


def _fade(seg: np.ndarray, sr: int, start: bool, end: bool) -> np.ndarray:
    n = min(int(FADE_S * sr), seg.size // 2)
    if n <= 1:
        return seg
    out = seg.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if start:
        out[:n] *= ramp
    if end:
        out[-n:] *= ramp[::-1]
    return out


def _runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    out, i = [], 0
    while i < mask.size:
        j = i
        while j < mask.size and mask[j] == mask[i]:
            j += 1
        out.append((bool(mask[i]), i, j))
        i = j
    return out


def wsola(x: np.ndarray, sr: int, factor: float, win_s: float = 0.020, tol_s: float = 0.010) -> np.ndarray:
    """Deterministic pitch-preserving time-scale modification (WSOLA).

    Output length ~= factor * input length. Hann windows at 50 % overlap; each next
    analysis frame is shifted within +/- ``tol_s`` to best continue the previous one
    (maximum cross-correlation), which preserves pitch periods without phase-vocoder
    smearing.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(round(win_s * sr)) // 2 * 2
    hs, tol = n // 2, int(round(tol_s * sr))
    ha = hs / factor
    out_len = int(round(x.size * factor))
    frames = out_len // hs + 2
    xp = np.pad(x, (tol, n + tol + int(np.ceil(frames * ha)) + 1))
    window = np.hanning(n + 2)[1:-1]
    y = np.zeros(frames * hs + n)
    wsum = np.zeros_like(y)
    delta = 0
    for k in range(frames):
        start = int(round(k * ha)) + tol + delta
        y[k * hs: k * hs + n] += xp[start: start + n] * window
        wsum[k * hs: k * hs + n] += window
        natural = xp[start + hs: start + hs + n]
        nominal = int(round((k + 1) * ha)) + tol
        region = xp[nominal - tol: nominal + tol + n]
        corr = np.correlate(region, natural, mode="valid")
        delta = int(np.argmax(corr)) - tol if corr.size and np.any(natural) else 0
    y = y / np.maximum(wsum, 1e-3)
    return y[:out_len].astype(np.float32)


def lengthen(x: np.ndarray, sr: int, factor: float, method: str = "wsola") -> np.ndarray:
    """Pitch-preserving lengthening.

    ``wsola`` (default) is deterministic. ``praat`` (PSOLA overlap-add) is not: Praat
    jitters pulses in unvoiced stretches, so repeated calls differ; use it only for
    listening comparisons, never for the dataset.
    """
    if factor <= 1.0001 or x.size < int(MIN_STRETCH_RUN_S * sr):
        return x.astype(np.float32)
    if method == "wsola":
        return wsola(x, sr, factor)
    if method != "praat":
        raise ValueError(f"unknown method {method!r}")
    parselmouth = af._parselmouth()
    from parselmouth.praat import call

    snd = parselmouth.Sound(x.astype(np.float64), sr)
    return call(snd, "Lengthen (overlap-add)", 75, 600, float(factor)).values[0].astype(np.float32)


class NoiseBed:
    """Background noise of the clip itself, looped to any length."""

    def __init__(self, x: np.ndarray, mask_samples: np.ndarray, sr: int, rng: np.random.Generator):
        self.sr, self.rng = sr, rng
        bed = x[~mask_samples]
        if bed.size < int(0.2 * sr):  # too little silence: synthesize at the clip's floor level
            floor_db = float(np.percentile(af.frame_levels_db(x, sr), 10))
            bed = (10 ** (floor_db / 20) * rng.standard_normal(int(0.5 * sr))).astype(np.float32)
            self.synthetic = True
        else:
            self.synthetic = False
        self.bed = bed.astype(np.float32)

    def take(self, n: int) -> np.ndarray:
        out = np.empty(0, dtype=np.float32)
        piece = max(int(0.1 * self.sr), min(self.bed.size, int(0.4 * self.sr)))
        while out.size < n:
            start = int(self.rng.integers(0, max(1, self.bed.size - piece + 1)))
            chunk = _fade(self.bed[start:start + piece], self.sr, True, True)
            out = np.concatenate([out, chunk])
        return out[:n]


def _draw_pause(params: TemporalParams, rng: np.random.Generator) -> float:
    for _ in range(100):
        d = float(np.exp(rng.normal(params.pause_mu, params.pause_sigma)))
        if MIN_PAUSE_S <= d <= params.max_pause_s:
            return d
    return float(np.clip(np.exp(params.pause_mu), MIN_PAUSE_S, params.max_pause_s))


def _insertion_points(seg: np.ndarray, sr: int) -> list[int]:
    """Sample offsets of local energy minima inside a (stretched) speech run."""
    db = af.frame_levels_db(seg, sr)
    hop = int(HOP_S * sr)
    guard = int(INSERT_EDGE_GUARD_S / HOP_S)
    if db.size <= 2 * guard + 2:
        return []
    smooth = np.convolve(db, np.ones(5) / 5, mode="same")
    cands = [i for i in range(guard, db.size - guard)
             if smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1] and smooth[i] < np.median(smooth) - 3]
    chosen: list[int] = []
    for i in sorted(cands, key=lambda k: smooth[k]):
        if all(abs(i - c) >= guard for c in chosen):
            chosen.append(i)
    return sorted(c * hop for c in chosen)


def transform(x: np.ndarray, sr: int, params: TemporalParams) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (audio, frame_labels, info). Deterministic for a given ``params.seed``."""
    rng = np.random.default_rng(params.seed)
    audio = np.asarray(x, dtype=np.float32)
    mask = af.energy_vad(audio, sr)
    hop = int(HOP_S * sr)
    mask_samples = np.repeat(mask, hop)[: audio.size]
    if mask_samples.size < audio.size:
        mask_samples = np.concatenate([mask_samples, np.zeros(audio.size - mask_samples.size, bool)])
    bed = NoiseBed(audio, mask_samples, sr, rng)

    runs = _runs(mask)
    speech_idx = [k for k, (is_sp, _, _) in enumerate(runs) if is_sp]
    if not speech_idx:
        raise ValueError("No speech detected; cannot apply temporal transform.")
    first, last = speech_idx[0], speech_idx[-1]

    # 1) stretch speech runs; collect internal pauses
    pieces: list[dict[str, Any]] = []
    for k, (is_sp, a, b) in enumerate(runs):
        seg = audio[a * hop: min(b * hop, audio.size)]
        if is_sp:
            pieces.append({"kind": "speech", "audio": lengthen(seg, sr, params.stretch), "src": (a * HOP_S, b * HOP_S)})
        else:
            kind = "pause" if first < k < last else "edge"
            pieces.append({"kind": kind, "audio": seg, "src": (a * HOP_S, b * HOP_S)})

    # 2) pause plan
    speech_out = sum(p["audio"].size for p in pieces if p["kind"] == "speech") / sr
    existing = [p for p in pieces if p["kind"] == "pause"]
    targets: dict[int, float] = {}
    for i, p in enumerate(existing):
        targets[i] = max(p["audio"].size / sr, _draw_pause(params, rng))
    inserts: list[float] = []
    for _ in range(4):  # fixed point: count depends on active duration, which depends on pauses
        active = speech_out + sum(targets.values()) + sum(inserts)
        n_target = int(round(params.pause_rate_per_min * active / 60.0))
        counted = sum(1 for d in targets.values() if d >= MIN_PAUSE_S)
        need = max(0, n_target - counted)
        while len(inserts) < need:
            inserts.append(_draw_pause(params, rng))
        inserts = inserts[:need]

    # 3) place inserted pauses at low-energy points in stretched speech runs
    slots = [(pi, off) for pi, p in enumerate(pieces) if p["kind"] == "speech" for off in _insertion_points(p["audio"], sr)]
    order = rng.permutation(len(slots)) if slots else []
    placed = sorted(slots[i] for i in order[: len(inserts)])
    by_piece: dict[int, list[int]] = {}
    for pi, off in placed:
        by_piece.setdefault(pi, []).append(off)

    # 4) assemble with labels
    out, labels = [], []
    draws = iter(inserts)

    def emit(seg: np.ndarray, label: int) -> None:
        out.append(seg.astype(np.float32))
        labels.append((label, seg.size))

    pause_i = 0
    for pi, p in enumerate(pieces):
        seg = p["audio"]
        if p["kind"] == "speech":
            cuts = by_piece.get(pi, [])
            bounds = [0, *cuts, seg.size]
            for j in range(len(bounds) - 1):
                part = seg[bounds[j]: bounds[j + 1]]
                emit(_fade(part, sr, j > 0, j < len(bounds) - 2), LABELS["speech"])
                if j < len(bounds) - 2:
                    emit(bed.take(int(next(draws) * sr)), LABELS["pause_inserted"])
        elif p["kind"] == "pause":
            extra = int(round((targets[pause_i] - seg.size / sr) * sr))
            pause_i += 1
            if extra > int(0.02 * sr):
                half = seg.size // 2
                emit(seg[:half], LABELS["pause"])
                emit(bed.take(extra), LABELS["pause_lengthened"])
                emit(seg[half:], LABELS["pause"])
            else:
                emit(seg, LABELS["pause"])
        else:
            emit(seg, LABELS["edge"])

    y = np.clip(np.concatenate(out), -1.0, 1.0).astype(np.float32)
    frame_labels = np.repeat([lab for lab, _ in labels], [n for _, n in labels])[::hop].astype(np.int8)
    info = {
        "params": asdict(params),
        "input_s": round(audio.size / sr, 3), "output_s": round(y.size / sr, 3),
        "speech_in_s": round(sum((b - a) * HOP_S for is_sp, a, b in runs if is_sp), 3),
        "speech_out_s": round(speech_out, 3),
        "existing_pauses": len(existing),
        "inserted_pauses": len(placed), "inserted_requested": len(inserts),
        "insertion_shortfall": len(inserts) - len(placed),
        "noise_bed_synthetic": bed.synthetic,
        "label_seconds": {name: round(float((frame_labels == v).sum() * HOP_S), 3) for name, v in LABELS.items()},
    }
    return y, frame_labels, info
