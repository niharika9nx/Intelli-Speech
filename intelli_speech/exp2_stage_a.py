"""Experiment 2 Stage A — measurement-validity logic (pure functions, testable offline).

Implements EXP2_PLAN.md v0.2.1 §8 and §13. Every threshold is pre-registered there and
duplicated here as a constant; changing one after seeing results violates the plan.
No group comparison (Stage B) and no synthetic articulation transforms live here
(the detectability perturbations are test stimuli only, §8.6).
"""

from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import articulation_features as ar
from .alignment import CORNER, CORNER_LONG, VOWEL_CATEGORIES, accepted_vowels

SR = 16000
# pre-registered gates (§8, §13)
SYNTH_MAX_MEDIAN_REL_ERR = 0.10
VALIDITY_MIN_RHO_VSA = 0.5
VALIDITY_MAX_RHO_FCR = -0.5
PURITY_MIN = 0.80
ICC_MIN = 0.6
CHANNEL_TOL_SD_FRACTION = 0.25
MANUAL_MEDIAN_ERR_MS = 30.0
MANUAL_WITHIN_50_MIN = 0.80
MANUAL_MAX_FAIL_RATE = 0.20
MANUAL_MIN_CATEGORY_AGREE = 0.85
MANUAL_N_UTTERANCES = 30
MANUAL_MIN_SPEAKERS = 10
MANUAL_MAX_PER_SPEAKER = 3
MANUAL_MIN_PER_CATEGORY = 3
MANUAL_MIN_LOW_CONF = 5
MANUAL_TOKENS_PER_UTT = 6
LEXICAL_MAX_SHARE_DIFF = 0.10
LEXICAL_MAX_JS = 0.10
MIN_CORNER_TOKENS = 3

H1 = ("aavs_log", "vsa_hull_log")
H23 = ("f2_excursion_st_median", "f2_speed_fit_oct_s")
H4 = ("pause_long_share_1s_win",)
APPLICABLE = {  # §13.2
    **{m: ("synthetic", "validity", "purity", "reliability", "channel", "detectability") for m in H1},
    **{m: ("synthetic", "purity", "reliability", "channel", "detectability") for m in H23},
    **{m: ("reliability", "channel") for m in H4},
}


def stable_hash(*parts: Any) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()


# --------------------------------------------------------------------------- synthetic recovery


def synth_vowel(f1: float, f2: float, f0: float, *, dur_s: float = 0.5, f3: float | None = None, seed: int = 0) -> np.ndarray:
    """Cascade formant vowel: glottal pulse train → 3 second-order resonators."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    n = int(dur_s * SR)
    src = np.zeros(n)
    period = SR / f0
    pos = 0.0
    while pos < n:
        src[int(pos)] = 1.0
        pos += period
    src = lfilter([1.0], [1.0, -0.97], src)            # -6 dB/oct glottal tilt (one pole)
    src = lfilter([1.0], [1.0, -0.90], src)
    f3 = f3 if f3 is not None else max(2500.0, f2 + 500.0)
    y = src
    for f, bw in ((f1, 80.0), (f2, 100.0), (f3, 150.0)):
        r = np.exp(-np.pi * bw / SR)
        theta = 2 * np.pi * f / SR
        a = [1.0, -2 * r * np.cos(theta), r * r]
        y = lfilter([1.0 - r], a, y)
    y = y / (np.abs(y).max() + 1e-12) * 0.5
    y = y + 1e-4 * rng.standard_normal(n)
    return y.astype(np.float32)


def synthetic_grid() -> list[tuple[float, float, float]]:
    out = []
    for f1 in (300.0, 450.0, 650.0, 800.0):
        for f2 in (900.0, 1300.0, 1800.0, 2300.0):
            if f2 < f1 + 300:
                continue
            for f0 in (100.0, 150.0, 200.0, 250.0, 300.0):
                out.append((f1, f2, f0))
    return out


def synthetic_recovery(grid: Iterable[tuple[float, float, float]] | None = None) -> dict[str, Any]:
    rows = []
    for f1, f2, f0 in (grid or synthetic_grid()):
        x = ar.harmonise(synth_vowel(f1, f2, f0))
        tr = ar.track(x, SR)
        t = tr.times
        mid = (t >= 0.2 * t[-1]) & (t <= 0.8 * t[-1])
        e1 = np.nanmedian(tr.f1[mid]) if mid.any() else np.nan
        e2 = np.nanmedian(tr.f2[mid]) if mid.any() else np.nan
        rows.append({"f1": f1, "f2": f2, "f0": f0, "f1_est": e1, "f2_est": e2,
                     "f1_rel_err": abs(e1 - f1) / f1 if np.isfinite(e1) else np.nan,
                     "f2_rel_err": abs(e2 - f2) / f2 if np.isfinite(e2) else np.nan})
    table = pd.DataFrame(rows)
    m1, m2 = float(table["f1_rel_err"].median()), float(table["f2_rel_err"].median())
    return {"table": table, "f1_median_rel_err": m1, "f2_median_rel_err": m2,
            "pass": bool(m1 <= SYNTH_MAX_MEDIAN_REL_ERR and m2 <= SYNTH_MAX_MEDIAN_REL_ERR)}


# --------------------------------------------------------------------------- reliability


def icc21(matrix: np.ndarray) -> float | None:
    """ICC(2,1): two-way random effects, absolute agreement, single measure. Rows = speakers."""
    m = np.asarray(matrix, dtype=float)
    m = m[np.all(np.isfinite(m), axis=1)]
    n, k = m.shape if m.ndim == 2 else (0, 0)
    if n < 3 or k < 2:
        return None
    grand = m.mean()
    ss_r = k * ((m.mean(axis=1) - grand) ** 2).sum()
    ss_c = n * ((m.mean(axis=0) - grand) ** 2).sum()
    ss_e = ((m - grand) ** 2).sum() - ss_r - ss_c
    ms_r, ms_c, ms_e = ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom > 0 else None


def split_half(features: pd.DataFrame, measure: str, salt: str = "exp2-split") -> dict[str, Any]:
    """Per speaker: recordings ordered by salted hash, alternating halves, median per half."""
    rows = []
    for spk, g in features.dropna(subset=[measure]).groupby("speaker_id"):
        if len(g) < 2:
            continue
        g = g.assign(_h=[stable_hash(salt, spk, r) for r in g["record_id"]]).sort_values("_h")
        a, b = g[measure].iloc[0::2], g[measure].iloc[1::2]
        rows.append([float(a.median()), float(b.median())])
    icc = icc21(np.array(rows)) if rows else None
    return {"n_speakers": len(rows), "icc21": icc}


# --------------------------------------------------------------------------- channel robustness


def _butter(x: np.ndarray, kind: str, hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    return sosfiltfilt(butter(8, hz, btype=kind, fs=SR, output="sos"), np.asarray(x, dtype=np.float64)).astype(np.float32)


def mulaw_8k(x: np.ndarray) -> np.ndarray:
    """8 kHz resampling + G.711 μ-law 8-bit round trip, back to 16 kHz."""
    from scipy.signal import resample_poly

    y = np.clip(resample_poly(np.asarray(x, dtype=np.float64), 1, 2), -1, 1)
    mu = 255.0
    enc = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round((enc + 1) / 2 * 255) / 255 * 2 - 1
    dec = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    return resample_poly(dec, 2, 1).astype(np.float32)[: x.size]


def ffmpeg_encoders() -> set[str]:
    exe = shutil.which("ffmpeg")
    if not exe:
        return set()
    try:
        out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return set()
    return {line.split()[1] for line in out.splitlines() if len(line.split()) > 1 and line.strip()[:1] in "AVS."}


def ffmpeg_roundtrip(x: np.ndarray, args: Sequence[str], ext: str) -> np.ndarray:
    import soundfile as sf

    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found")
    with tempfile.TemporaryDirectory() as d:
        src, enc, dec = Path(d, "in.wav"), Path(d, f"enc.{ext}"), Path(d, "out.wav")
        sf.write(src, np.asarray(x, dtype=np.float32), SR, subtype="PCM_16")
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src), *args, str(enc)], check=True, timeout=120)
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(enc), "-ar", str(SR), "-ac", "1", str(dec)], check=True, timeout=120)
        y, _ = sf.read(dec, dtype="float32")
    return y[: x.size] if y.size >= x.size else np.pad(y, (0, x.size - y.size))


def channel_conditions(encoders: set[str] | None = None) -> dict[str, Callable[[np.ndarray], np.ndarray] | None]:
    """Condition name → transform, or None when unavailable (reported as 'not run')."""
    enc = ffmpeg_encoders() if encoders is None else encoders
    conds: dict[str, Callable | None] = {
        "lp3000": lambda x: _butter(x, "lowpass", 3000.0),
        "lp3400": lambda x: _butter(x, "lowpass", 3400.0),
        "lp4000": lambda x: _butter(x, "lowpass", 4000.0),
        "hp300": lambda x: _butter(x, "highpass", 300.0),
        "g711_mulaw_8k": mulaw_8k,
    }
    opus = "libopus" in enc
    conds["opus_8k"] = (lambda x: ffmpeg_roundtrip(x, ["-c:a", "libopus", "-b:a", "8k"], "ogg")) if opus else None
    conds["opus_12k"] = (lambda x: ffmpeg_roundtrip(x, ["-c:a", "libopus", "-b:a", "12k"], "ogg")) if opus else None
    amr = "libopencore_amrnb" in enc
    conds["amrnb_12k2"] = (lambda x: ffmpeg_roundtrip(x, ["-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k"], "amr")) if amr else None
    return conds


def channel_gate(original: pd.DataFrame, conditioned: dict[str, pd.DataFrame], measure: str) -> dict[str, Any]:
    """§13.4: shift = median over recordings (cond − orig); pass if |shift| < 0.25 × SD(speaker medians)."""
    spk = original.dropna(subset=[measure]).groupby("speaker_id")[measure].median()
    sd = float(spk.std(ddof=1)) if spk.size >= 3 else float("nan")
    tol = CHANNEL_TOL_SD_FRACTION * sd
    base = original.set_index("record_id")[measure]
    out: dict[str, Any] = {"tolerance": tol, "between_speaker_sd": sd, "conditions": {}}
    status = "pass" if np.isfinite(tol) else "incomplete"
    for name, frame in conditioned.items():
        if frame is None:
            out["conditions"][name] = {"status": "not_run"}
            status = "incomplete" if status == "pass" else status
            continue
        joined = pd.concat([base.rename("orig"), frame.set_index("record_id")[measure].rename("cond")], axis=1).dropna()
        if joined.empty:
            out["conditions"][name] = {"status": "no_data"}
            status = "incomplete" if status == "pass" else status
            continue
        shift = float((joined["cond"] - joined["orig"]).median())
        ok = bool(np.isfinite(tol) and abs(shift) < tol)
        out["conditions"][name] = {"status": "pass" if ok else "fail", "shift": shift, "n": int(len(joined))}
        if not ok:
            status = "fail"
    out["status"] = status
    return out


# --------------------------------------------------------------------------- labelled measures, purity, validity


def token_formants(tr: ar.Tracks, start_s: float, end_s: float) -> tuple[float, float] | None:
    """Median ln F1/ln F2 over the central 40 % of a token (range+bandwidth filters only)."""
    d = end_s - start_s
    a, b = start_s + 0.3 * d, end_s - 0.3 * d
    ok = (tr.times >= a) & (tr.times <= b)
    ok &= np.isfinite(tr.f1) & np.isfinite(tr.f2) & np.isfinite(tr.b1) & np.isfinite(tr.b2)
    ok &= (tr.f1 >= ar.F1_RANGE[0]) & (tr.f1 <= ar.F1_RANGE[1]) & (tr.f2 >= ar.F2_RANGE[0]) & (tr.f2 <= ar.F2_RANGE[1])
    ok &= (tr.b1 < ar.B1_MAX) & (tr.b2 < ar.B2_MAX)
    if not ok.any():
        return None
    return float(np.median(np.log(tr.f1[ok]))), float(np.median(np.log(tr.f2[ok])))


def labelled_speaker_measures(tokens: pd.DataFrame, corner: dict[str, tuple[str, ...]] = CORNER) -> pd.DataFrame:
    """tokens: speaker_id, category, lf1, lf2 → FCR (Hz) and log-VSA per speaker with ≥3 tokens per corner."""
    rows = []
    for spk, g in tokens.dropna(subset=["lf1", "lf2"]).groupby("speaker_id"):
        means = {}
        for c, cats in corner.items():
            sub = g[g["category"].isin(cats)]
            if len(sub) < MIN_CORNER_TOKENS:
                break
            means[c] = (float(sub["lf1"].mean()), float(sub["lf2"].mean()))
        if len(means) < 3:
            rows.append({"speaker_id": spk, "fcr": None, "log_vsa": None, "sufficient": False})
            continue
        (i1, i2), (a1, a2), (u1, u2) = means["i"], means["a"], means["u"]
        F = {k: (np.exp(v[0]), np.exp(v[1])) for k, v in means.items()}
        fcr = (F["u"][1] + F["a"][1] + F["i"][0] + F["u"][0]) / (F["i"][1] + F["a"][0])
        area = 0.5 * abs(i1 * (a2 - u2) + a1 * (u2 - i2) + u1 * (i2 - a2))
        rows.append({"speaker_id": spk, "fcr": float(fcr), "log_vsa": float(area), "sufficient": True})
    return pd.DataFrame(rows, columns=["speaker_id", "fcr", "log_vsa", "sufficient"])


def frame_purity(selected_times: np.ndarray, vowel_spans: Sequence[tuple[float, float]]) -> tuple[int, int]:
    """(selected frames inside any aligned vowel, selected frames total)."""
    t = np.asarray(selected_times)
    if t.size == 0:
        return 0, 0
    inside = np.zeros(t.size, bool)
    for a, b in vowel_spans:
        inside |= (t >= a) & (t <= b)
    return int(inside.sum()), int(t.size)


def validity_gate(label_free: pd.DataFrame, labelled: pd.DataFrame, measure: str) -> dict[str, Any]:
    """Speaker-level Spearman: ρ(measure, log_vsa) ≥ 0.5 and ρ(measure, fcr) ≤ −0.5."""
    lab = labelled[labelled["sufficient"]]
    j = label_free.groupby("speaker_id")[measure].median().rename("m").to_frame().join(lab.set_index("speaker_id"), how="inner").dropna()
    n = len(j)
    if n < 5:
        return {"status": "incomplete", "n_speakers": n, "reason": "fewer than 5 speakers with labelled measures"}
    rv = float(j["m"].rank().corr(j["log_vsa"].rank()))
    rf = float(j["m"].rank().corr(j["fcr"].rank()))
    ok = rv >= VALIDITY_MIN_RHO_VSA and rf <= VALIDITY_MAX_RHO_FCR
    return {"status": "pass" if ok else "fail", "n_speakers": n, "rho_log_vsa": rv, "rho_fcr": rf}


# --------------------------------------------------------------------------- manual validation protocol (§13.3)


def select_manual_sample(alignments: pd.DataFrame, salt: str = "exp2-manual") -> pd.DataFrame:
    """alignments: utt_id, speaker_id, utterance_conf, status, categories (list of vowel categories).

    Deterministic: ≥10 speakers (≤3 each), each category in ≥3 utterances where available,
    ≥5 from the lowest-confidence quartile.
    """
    cand = alignments[alignments["status"].isin(["aligned", "low_confidence"])].copy()
    if cand.empty:
        return cand.assign(selection_reason=[])
    cand["_h"] = [stable_hash(salt, u) for u in cand["utt_id"]]
    cand["_cats"] = cand["categories"].apply(lambda c: set(c))
    q25 = cand["utterance_conf"].quantile(0.25)
    cand["_low"] = cand["utterance_conf"] <= q25
    cand = cand.sort_values("_h")
    chosen: list[int] = []
    reasons: dict[int, str] = {}
    per_spk: dict[str, int] = {}
    cover = {c: 0 for c in VOWEL_CATEGORIES}

    def take(idx: int, why: str) -> None:
        chosen.append(idx)
        reasons[idx] = why
        spk = cand.at[idx, "speaker_id"]
        per_spk[spk] = per_spk.get(spk, 0) + 1
        for c in cand.at[idx, "_cats"]:
            if c in cover:
                cover[c] += 1

    def gain(idx: int) -> int:
        return sum(1 for c in cand.at[idx, "_cats"] if c in cover and cover[c] < MANUAL_MIN_PER_CATEGORY)

    # 1) one per speaker for the first speakers (max coverage gain)
    for spk, g in cand.groupby("speaker_id", sort=False):
        if len(per_spk) >= MANUAL_MIN_SPEAKERS:
            break
        best = max(g.index, key=lambda i: (gain(i), -int(cand.at[i, "_h"], 16) % 997))
        take(best, "speaker_coverage")
    # 2) low-confidence quota
    for idx in cand[cand["_low"]].index:
        if sum(1 for i in chosen if cand.at[i, "_low"]) >= MANUAL_MIN_LOW_CONF or len(chosen) >= MANUAL_N_UTTERANCES:
            break
        if idx not in chosen and per_spk.get(cand.at[idx, "speaker_id"], 0) < MANUAL_MAX_PER_SPEAKER:
            take(idx, "low_confidence_quota")
    # 3) category coverage, then fill
    while len(chosen) < MANUAL_N_UTTERANCES:
        pool = [i for i in cand.index if i not in chosen and per_spk.get(cand.at[i, "speaker_id"], 0) < MANUAL_MAX_PER_SPEAKER]
        if not pool:
            break
        best = max(pool, key=lambda i: (gain(i), len(cand.at[i, "_cats"])))
        take(best, "category_coverage" if gain(best) else "fill")
    out = cand.loc[chosen].drop(columns=["_h", "_cats"]).rename(columns={"_low": "low_conf_quartile"})
    out["selection_reason"] = [reasons[i] for i in chosen]
    out.attrs["category_coverage"] = cover
    return out


def select_manual_tokens(vowels_by_utt: dict[str, list[dict[str, Any]]], utt_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Up to 6 tokens per utterance, preferring categories with the fewest selected tokens so far."""
    counts = {c: 0 for c in VOWEL_CATEGORIES}
    rows = []
    for u in utt_ids:
        toks = [dict(v, token_index=i) for i, v in enumerate(vowels_by_utt.get(u, []))]
        picked = []
        while toks and len(picked) < MANUAL_TOKENS_PER_UTT:
            best = min(toks, key=lambda v: (counts.get(v["category"], 0), v["token_index"]))
            picked.append(best)
            toks.remove(best)
            counts[best["category"]] = counts.get(best["category"], 0) + 1
        rows += [{"utt_id": u, **p} for p in sorted(picked, key=lambda v: v["token_index"])]
    return rows


def _num(v: Any) -> float | None:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def summarise_manual(tokens: pd.DataFrame, utterances: pd.DataFrame) -> dict[str, Any]:
    """tokens sheet: auto_start_s, auto_end_s, auto_conf, category, human_start_s ('=' or s), human_end_s,
    vowel_present (y/n), category_correct (y/n). utterances sheet: gross_misalignment (y/n)."""
    errs, rows = [], []
    for _, r in tokens.iterrows():
        hs, he = str(r.get("human_start_s", "")).strip(), str(r.get("human_end_s", "")).strip()
        if hs in ("", "nan") or he in ("", "nan"):
            continue                                   # not checked
        present = str(r.get("vowel_present", "")).strip().lower() == "y"
        row = {"category": r["category"], "long": r["category"] in ("aa", "ii", "uu", "ee", "oo"),
               "conf": _num(r.get("auto_conf")), "present": present,
               "category_correct": str(r.get("category_correct", "")).strip().lower() == "y"}
        if present:
            s = _num(r["auto_start_s"]) if hs == "=" else _num(hs)
            e = _num(r["auto_end_s"]) if he == "=" else _num(he)
            if s is not None and e is not None:
                es = abs(s - float(r["auto_start_s"])) * 1000
                ee = abs(e - float(r["auto_end_s"])) * 1000
                row.update({"err_start_ms": es, "err_end_ms": ee})
                errs += [es, ee]
        rows.append(row)
    t = pd.DataFrame(rows)
    checked_utts = utterances[utterances["gross_misalignment"].astype(str).str.strip().str.lower().isin(["y", "n"])]
    fail_rate = float((checked_utts["gross_misalignment"].astype(str).str.lower().str.strip() == "y").mean()) if len(checked_utts) else None
    e = np.array(errs)
    present = t[t["present"]] if len(t) else t
    agree = float(present["category_correct"].mean()) if len(present) else None
    by = {}
    if "err_start_ms" in t:
        t["mean_err"] = t[["err_start_ms", "err_end_ms"]].mean(axis=1)
        for key, g in [("short", t[~t["long"]]), ("long", t[t["long"]])] + [(c, t[t["category"] == c]) for c in VOWEL_CATEGORIES]:
            ge = g[["err_start_ms", "err_end_ms"]].to_numpy().ravel()
            ge = ge[np.isfinite(ge)]
            by[key] = {"n_boundaries": int(ge.size), "median_ms": float(np.median(ge)) if ge.size else None,
                       "within_50": float((ge <= 50).mean()) if ge.size else None}
        cc = t[["conf", "mean_err"]].dropna()
        rho = float(cc["conf"].rank().corr(cc["mean_err"].rank())) if len(cc) >= 5 else None
    else:
        rho = None
    summary = {
        "n_tokens_checked": int(len(t)), "n_boundaries": int(e.size), "n_utterances_checked": int(len(checked_utts)),
        "median_abs_boundary_ms": float(np.median(e)) if e.size else None,
        "p90_abs_boundary_ms": float(np.percentile(e, 90)) if e.size else None,
        "within_20ms": float((e <= 20).mean()) if e.size else None,
        "within_50ms": float((e <= 50).mean()) if e.size else None,
        "utterance_failure_rate": fail_rate, "category_agreement": agree,
        "spearman_conf_vs_error": rho, "by_group": by,
    }
    complete = e.size > 0 and fail_rate is not None and agree is not None
    ok = complete and (summary["median_abs_boundary_ms"] <= MANUAL_MEDIAN_ERR_MS and summary["within_50ms"] >= MANUAL_WITHIN_50_MIN
                       and fail_rate <= MANUAL_MAX_FAIL_RATE and agree >= MANUAL_MIN_CATEGORY_AGREE)
    summary["acceptance"] = "pass" if ok else ("fail" if complete else "incomplete")
    return summary


# --------------------------------------------------------------------------- lexical comparability (§13.4)


def corner_shares(counts: dict[str, int]) -> dict[str, float]:
    tot = {c: sum(counts.get(x, 0) for x in cats) for c, cats in CORNER.items()}
    s = sum(tot.values())
    return {c: (v / s if s else 0.0) for c, v in tot.items()}


def lexical_flag(control_counts: dict[str, int], atypical_counts: dict[str, int]) -> dict[str, Any]:
    p, q = corner_shares(control_counts), corner_shares(atypical_counts)
    keys = list(CORNER)
    pv, qv = np.array([p[k] for k in keys]), np.array([q[k] for k in keys])
    m = 0.5 * (pv + qv)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    js = float(np.sqrt(0.5 * kl(pv, m) + 0.5 * kl(qv, m))) if pv.sum() and qv.sum() else None
    maxdiff = float(np.max(np.abs(pv - qv)))
    flagged = bool(maxdiff > LEXICAL_MAX_SHARE_DIFF or (js is not None and js > LEXICAL_MAX_JS))
    return {"control_shares": p, "atypical_shares": q, "max_share_diff": maxdiff, "js_distance": js, "flag": flagged}


# --------------------------------------------------------------------------- detectability (§8.6, optional pyworld)


def envelope_perturbation(x: np.ndarray, kind: str, level: float) -> np.ndarray:
    """TEST STIMULUS ONLY. kind='warp': interpolate log envelope toward its time mean by `level`;
    kind='smooth': moving-average log envelope over `level` seconds. Requires pyworld."""
    import pyworld as pw

    xd = np.asarray(x, dtype=np.float64)
    f0, t = pw.harvest(xd, SR, frame_period=5.0)
    sp = pw.cheaptrick(xd, f0, t, SR)
    ap = pw.d4c(xd, f0, t, SR)
    lsp = np.log(sp + 1e-12)
    if kind == "warp":
        lsp = (1 - level) * lsp + level * lsp.mean(axis=0, keepdims=True)
    elif kind == "smooth":
        k = max(1, int(round(level / 0.005)))
        kernel = np.ones(k) / k
        lsp = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, lsp)
    else:
        raise ValueError(kind)
    y = pw.synthesize(f0, np.exp(lsp), ap, SR, frame_period=5.0)
    return y[: xd.size].astype(np.float32)


def monotonic_decrease(values: Sequence[float | None]) -> bool:
    v = [x for x in values]
    if any(x is None or not np.isfinite(x) for x in v):
        return False
    return all(b < a for a, b in zip(v[:-1], v[1:]))


# --------------------------------------------------------------------------- speaker aggregation + report


def speaker_sufficiency(features: pd.DataFrame) -> pd.DataFrame:
    """Per speaker: eligible clips (gate pass) and selected frames; formant sufficiency flag."""
    g = features.groupby("speaker_id")
    out = pd.DataFrame({
        "n_clips": g.size(),
        "n_gate_clips": g["gate_pass"].sum(),
        "n_selected_frames": g["sel_n_selected"].sum() if "sel_n_selected" in features else 0,
    })
    out["formant_sufficient"] = (out["n_gate_clips"] >= ar.MIN_CLIPS_SPEAKER) & (out["n_selected_frames"] >= ar.MIN_FRAMES_SPEAKER)
    return out.reset_index()


def measure_status(results: dict[str, dict[str, Any]], measure: str) -> dict[str, Any]:
    """Combine per-test statuses ('pass'|'fail'|'incomplete') for one measure (§13.2)."""
    tests = {name: results.get(name, {}).get(measure, {"status": "incomplete"}) for name in APPLICABLE[measure]}
    statuses = [t.get("status", "incomplete") for t in tests.values()]
    overall = "fail" if "fail" in statuses else ("incomplete" if "incomplete" in statuses else "pass")
    return {"measure": measure, "overall": overall, "tests": tests}
