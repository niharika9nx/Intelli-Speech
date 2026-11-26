from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("parselmouth")
pytest.importorskip("scipy")

from intelli_speech import articulation_features as ar
from intelli_speech.exp2_stage_a import synth_vowel

SR = 16000


def _noise(seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    return (0.1 * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)


def _lowpass(x: np.ndarray, hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    return sosfiltfilt(butter(10, hz, fs=SR, output="sos"), x).astype(np.float32)


def _vowel_sentence(seed: int = 0) -> np.ndarray:
    """Synthetic 'utterance': vowels separated by short silences and one long pause, plus
    breath-level broadband noise (pure resonator vowels have no energy above ~2.2 kHz)."""
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(0.3 * SR), np.float32)]
    targets = [(300, 2300), (700, 1300), (350, 900), (500, 1800), (650, 1100), (300, 2100)]
    for k, (f1, f2) in enumerate(targets * 2):
        parts.append(synth_vowel(f1, f2, 140 + 10 * (k % 3), dur_s=0.18, seed=k))
        parts.append(np.zeros(int((1.2 if k == 5 else 0.12) * SR), np.float32))
    x = np.concatenate(parts)
    return (x + 0.01 * rng.standard_normal(x.size)).astype(np.float32)


# --------------------------------------------------------------------------- channel conditioning


def test_cutoff_descriptor_tracks_band_limit_and_handles_silence() -> None:
    wide = ar.channel_descriptors(_noise())
    narrow = ar.channel_descriptors(_lowpass(_noise(), 3000.0))
    assert wide["upper_cutoff_hz"] >= 7000
    assert 2600 <= narrow["upper_cutoff_hz"] <= 3600
    assert ar.channel_descriptors(np.zeros(SR, np.float32)) == {"upper_cutoff_hz": None, "low_ratio_db": None}
    assert ar.channel_descriptors(np.ones(100, np.float32))["upper_cutoff_hz"] is None   # too short
    boosted = ar.channel_descriptors(_lowpass(_noise(), 250.0) + 0.01 * _noise(seed=1))
    assert boosted["low_ratio_db"] > wide["low_ratio_db"]


def test_harmonise_band_limits_and_rejects_wrong_rate() -> None:
    t = np.arange(2 * SR) / SR
    def tone(f):
        return np.sin(2 * np.pi * f * t).astype(np.float32)
    def gain_db(f):
        return 20 * np.log10(np.std(ar.harmonise(tone(f))[SR // 2: -SR // 2]) / np.std(tone(f)))
    assert gain_db(1000) > -1.0
    assert gain_db(40) < -30 and gain_db(6000) < -30
    with pytest.raises(ValueError):
        ar.harmonise(tone(1000), sr=8000)
    assert ar.harmonise(np.zeros(10, np.float32)).size == 10   # tiny input passes through


# --------------------------------------------------------------------------- vowel space


def test_aavs_log_is_scale_invariant_and_sensitive_to_centralisation() -> None:
    rng = np.random.default_rng(1)
    lf1, lf2 = np.log(rng.uniform(300, 800, 400)), np.log(rng.uniform(900, 2300, 400))
    base = ar.aavs_log(lf1, lf2)
    assert ar.aavs_log(lf1 + np.log(1.18), lf2 + np.log(1.18)) == pytest.approx(base, rel=1e-9)
    shrunk = ar.aavs_log(lf1.mean() + 0.6 * (lf1 - lf1.mean()), lf2.mean() + 0.6 * (lf2 - lf2.mean()))
    assert shrunk == pytest.approx(0.36 * base, rel=1e-6)          # area-like: scales with factor^2
    assert ar.aavs_log(lf1[:10], lf2[:10]) is None                 # missing data → None, not 0


def test_vsa_hull_handles_degenerate_clouds() -> None:
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, 300)
    assert ar.vsa_hull_log(a, a) == 0.0                            # collinear: no crash
    square = ar.vsa_hull_log(rng.uniform(0, 1, 500), rng.uniform(0, 1, 500))
    assert 0.5 < square < 1.0                                      # trimmed unit square
    assert ar.vsa_hull_log(a[:5], a[:5]) is None


# --------------------------------------------------------------------------- dynamics


def _tracks(f2: np.ndarray) -> ar.Tracks:
    n = f2.size
    t = np.arange(n) * ar.HOP_S + 0.0125
    return ar.Tracks(times=t, f1=np.full(n, 500.0), f2=f2, b1=np.full(n, 80.0), b2=np.full(n, 120.0),
                     voiced=np.ones(n, bool), intensity_db=np.full(n, 60.0), nasal=np.zeros(n, bool))


def test_formant_dynamics_measures_excursion_and_speed() -> None:
    n = 120
    f2 = 1200.0 * 2 ** (np.linspace(0, 1, n))        # one octave rise over 1.2 s → 12 st
    tr = _tracks(f2)
    nuclei = np.array([0.1, 0.4, 0.7, 1.0])
    d = ar.formant_dynamics(tr, nuclei)
    assert d["n_transition_pairs"] == 3
    assert d["f2_excursion_st_median"] == pytest.approx(3.0, abs=0.35)   # 0.3 s of a 12 st/1.19 s rise
    assert d["f2_speed_fit_oct_s"] == pytest.approx(1 / 1.19, rel=0.15)
    flat = ar.formant_dynamics(_tracks(np.full(n, 1500.0)), nuclei)
    assert flat["f2_excursion_st_median"] == pytest.approx(0.0, abs=1e-6)
    assert flat["f2_speed_fit_oct_s"] is None                   # no transition ≥ 1 st
    assert ar.formant_dynamics(tr, np.array([0.1]))["f2_excursion_st_median"] is None


def test_dct_fit_smooths_noise_but_keeps_trend() -> None:
    rng = np.random.default_rng(3)
    y = np.linspace(0, 1, 50) + 0.05 * rng.standard_normal(50)
    fit = ar.dct_fit(y)
    assert np.abs(fit - np.linspace(0, 1, 50)).max() < 0.08
    assert ar.dct_fit(np.array([1.0, 2.0])).size == 2           # fewer points than coefficients


# --------------------------------------------------------------------------- pauses and EMS


def test_pause_windows_count_only_contained_pauses() -> None:
    mask = np.ones(1200, bool)                                   # 12 s
    mask[100:115] = False                                        # 0.15 s pause in window 1
    mask[300:430] = False                                        # 1.3 s long pause in window 1
    mask[570:600] = False                                        # straddles 5.8 s boundary → excluded
    mask[700:705] = False                                        # 0.05 s → too short
    w = ar.pause_windows(mask, 12.0)
    assert w["n_windows"] == 2
    assert w["pause_count_win"] == 2 and w["pause_long_count_win"] == 1
    assert w["pause_long_share_1s_win"] == pytest.approx(0.5)
    assert ar.pause_windows(np.ones(300, bool), 3.0)["n_windows"] == 0
    short = ar.pause_windows(np.ones(450, bool), 4.5)
    assert short["n_windows"] == 1 and short["pause_long_share_1s_win"] is None
    assert ar.pause_windows(np.zeros(800, bool), 8.0)["pause_count_win"] == 0   # no speech at all


def test_ems_finds_modulation_rate() -> None:
    t = np.arange(6 * SR) / SR
    am = (1 + 0.9 * np.sin(2 * np.pi * 4.0 * t)) * _noise(6.0)
    e = ar.ems(am)
    assert e["peak_hz"] == pytest.approx(4.0, abs=0.3) and e["ratio_3_5"] > 0.5
    assert ar.ems(am[: SR])["ratio_3_5"] is None                # shorter than 2 s
    assert ar.ems(np.zeros(3 * SR))["ratio_3_5"] is None


# --------------------------------------------------------------------------- end to end


def test_extract_on_vowel_sentence_and_gate() -> None:
    x = _vowel_sentence()
    f = ar.extract(x)
    assert f["feature_version"] == ar.FEATURE_VERSION and f["gate_pass"]
    assert f["sel_n_selected"] > 0 and f["nuclei_per_s"] > 0
    assert f["f0_median_hz"] == pytest.approx(150, rel=0.1)
    narrow = ar.extract(_lowpass(x, 2500.0) + 1e-4 * _noise(x.size / SR))
    assert narrow["gate_pass"] is False
    assert narrow["aavs_log"] is None and narrow["f2_excursion_st_median"] is None
    assert narrow["n_windows"] >= 1                              # timing still measured


def test_extract_missing_data_edge_cases() -> None:
    with pytest.raises(ValueError):
        ar.extract(np.zeros(100, np.float32))
    with pytest.raises(ValueError):
        ar.extract(np.zeros((2, SR), np.float32))
    silent = ar.extract(np.zeros(2 * SR, np.float32))           # must not crash
    assert silent["aavs_log"] is None and silent["upper_cutoff_hz"] is None and silent["gate_pass"] is False
    for key in ar.PRIMARY_MEASURES:
        assert key in silent
