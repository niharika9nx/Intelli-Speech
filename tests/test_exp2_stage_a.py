from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intelli_speech import exp2_stage_a as sa
from intelli_speech.alignment import VOWEL_CATEGORIES

SR = 16000


# --------------------------------------------------------------------------- reliability


def test_icc21_agreement_noise_and_missing() -> None:
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 1, 40)
    good = np.column_stack([truth + 0.05 * rng.normal(size=40), truth + 0.05 * rng.normal(size=40)])
    assert sa.icc21(good) > 0.95
    noise = rng.normal(0, 1, (40, 2))
    assert abs(sa.icc21(noise)) < 0.4
    assert sa.icc21(np.array([[1.0, 2.0], [2.0, 3.0]])) is None            # < 3 speakers
    with_nan = good.copy()
    with_nan[0, 0] = np.nan
    assert sa.icc21(with_nan) == pytest.approx(sa.icc21(good[1:]))         # NaN rows dropped


def test_split_half_uses_alternating_halves_and_skips_single_clip_speakers() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for s in range(20):
        level = rng.normal()
        for r in range(4):
            rows.append({"speaker_id": f"s{s}", "record_id": f"s{s}r{r}", "m": level + 0.1 * rng.normal()})
    rows.append({"speaker_id": "lonely", "record_id": "x", "m": 5.0})
    rows.append({"speaker_id": "empty", "record_id": "y", "m": None})
    out = sa.split_half(pd.DataFrame(rows), "m")
    assert out["n_speakers"] == 20 and out["icc21"] > 0.8


# --------------------------------------------------------------------------- channel


def _features(values: dict[str, float], speakers: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{"record_id": r, "speaker_id": speakers[r], "m": v} for r, v in values.items()])


def test_channel_gate_pass_fail_and_incomplete() -> None:
    speakers = {f"r{i}": f"s{i // 2}" for i in range(20)}
    orig = _features({f"r{i}": float(i // 2) for i in range(20)}, speakers)     # speaker medians 0..9, SD≈3.03
    small = _features({f"r{i}": float(i // 2) + 0.2 for i in range(20)}, speakers)
    big = _features({f"r{i}": float(i // 2) + 2.0 for i in range(20)}, speakers)
    res = sa.channel_gate(orig, {"lp": small}, "m")
    assert res["status"] == "pass" and res["tolerance"] == pytest.approx(0.25 * np.std(range(10), ddof=1))
    assert sa.channel_gate(orig, {"lp": small, "hp": big}, "m")["status"] == "fail"
    inc = sa.channel_gate(orig, {"lp": small, "opus_8k": None}, "m")
    assert inc["status"] == "incomplete" and inc["conditions"]["opus_8k"]["status"] == "not_run"
    empty = _features({f"r{i}": np.nan for i in range(20)}, speakers)
    assert sa.channel_gate(orig, {"lp": empty}, "m")["status"] == "incomplete"


def test_numeric_channel_conditions_and_codec_availability() -> None:
    pytest.importorskip("scipy")
    x = (0.1 * np.random.default_rng(2).standard_normal(SR)).astype(np.float32)
    y = sa.mulaw_8k(x)
    assert y.size == x.size
    spec = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(y.size, 1 / SR)
    assert spec[freqs > 4500].sum() < 0.01 * spec.sum()                    # 8 kHz path removes > 4 kHz
    conds = sa.channel_conditions(encoders=set())
    assert conds["opus_8k"] is None and conds["amrnb_12k2"] is None
    assert callable(conds["lp3400"]) and callable(conds["hp300"])
    assert callable(sa.channel_conditions(encoders={"libopus"})["opus_12k"])
    hp = conds["hp300"](x)
    spec_hp = np.abs(np.fft.rfft(hp)) ** 2
    assert spec_hp[freqs < 150].sum() < 0.01 * spec_hp[(freqs > 500) & (freqs < 650)].sum()


# --------------------------------------------------------------------------- labelled measures, purity, validity


def _tokens(spk: str, scale: float, n: int = 4) -> list[dict]:
    corner = {"i": (300, 2300), "a": (750, 1300), "u": (320, 800)}
    rows = []
    for cat, (f1, f2) in corner.items():
        c1, c2 = np.log(500), np.log(1400)
        for k in range(n):
            rows.append({"speaker_id": spk, "category": cat if k % 2 else cat * 2 if cat != "a" else "aa",
                         "lf1": c1 + scale * (np.log(f1) - c1), "lf2": c2 + scale * (np.log(f2) - c2)})
    return rows


def test_labelled_measures_fcr_and_vsa_respond_to_centralisation() -> None:
    tokens = pd.DataFrame(_tokens("wide", 1.0) + _tokens("narrow", 0.5) + _tokens("thin", 1.0, n=2))
    out = sa.labelled_speaker_measures(tokens).set_index("speaker_id")
    assert out.loc["wide", "log_vsa"] > out.loc["narrow", "log_vsa"] > 0
    assert out.loc["wide", "fcr"] < out.loc["narrow", "fcr"]               # centralisation raises FCR
    assert not out.loc["thin", "sufficient"]                                 # < 3 tokens per corner
    expected = (800 + 1300 + 300 + 320) / (2300 + 750)
    assert out.loc["wide", "fcr"] == pytest.approx(expected, rel=1e-9)


def test_frame_purity_counts() -> None:
    assert sa.frame_purity(np.array([0.1, 0.2, 0.5, 0.9]), [(0.05, 0.25), (0.85, 1.0)]) == (3, 4)
    assert sa.frame_purity(np.array([]), [(0, 1)]) == (0, 0)


def test_validity_gate_pass_fail_incomplete() -> None:
    rng = np.random.default_rng(3)
    spk = [f"s{i}" for i in range(12)]
    vsa = rng.uniform(0.5, 2.0, 12)
    labelled = pd.DataFrame({"speaker_id": spk, "log_vsa": vsa, "fcr": 2.0 - 0.3 * vsa, "sufficient": True})
    lf = pd.DataFrame({"speaker_id": spk, "aavs_log": vsa + 0.05 * rng.normal(size=12)})
    assert sa.validity_gate(lf, labelled, "aavs_log")["status"] == "pass"
    lf_bad = lf.assign(aavs_log=rng.permutation(lf["aavs_log"].to_numpy()) * 0 + rng.normal(size=12))
    assert sa.validity_gate(lf_bad, labelled, "aavs_log")["status"] == "fail"
    assert sa.validity_gate(lf.head(3), labelled, "aavs_log")["status"] == "incomplete"
    insufficient = labelled.assign(sufficient=False)
    assert sa.validity_gate(lf, insufficient, "aavs_log")["status"] == "incomplete"


# --------------------------------------------------------------------------- manual validation protocol


def _alignments(n_speakers: int = 20, per: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = []
    for s in range(n_speakers):
        for u in range(per):
            cats = list(rng.choice(VOWEL_CATEGORIES, size=rng.integers(3, 7), replace=False))
            rows.append({"utt_id": f"s{s}u{u}", "speaker_id": f"s{s}", "utterance_conf": float(rng.uniform(0.3, 0.95)),
                         "status": "aligned", "categories": cats})
    rows.append({"utt_id": "fail", "speaker_id": "s0", "utterance_conf": 0.0, "status": "failed", "categories": []})
    return pd.DataFrame(rows)


def test_manual_sample_meets_protocol_constraints_and_is_deterministic() -> None:
    al = _alignments()
    sample = sa.select_manual_sample(al)
    again = sa.select_manual_sample(al.sample(frac=1.0, random_state=9))
    assert list(sample["utt_id"]) == list(again["utt_id"])                    # order-independent
    assert len(sample) == sa.MANUAL_N_UTTERANCES
    assert sample["speaker_id"].nunique() >= sa.MANUAL_MIN_SPEAKERS
    assert sample.groupby("speaker_id").size().max() <= sa.MANUAL_MAX_PER_SPEAKER
    assert sample["low_conf_quartile"].sum() >= sa.MANUAL_MIN_LOW_CONF
    assert "fail" not in set(sample["utt_id"])
    for c in VOWEL_CATEGORIES:
        assert sum(c in cats for cats in sample["categories"]) >= sa.MANUAL_MIN_PER_CATEGORY


def test_manual_sample_with_few_speakers_respects_caps() -> None:
    sample = sa.select_manual_sample(_alignments(n_speakers=4, per=10))
    assert len(sample) == 12 and sample.groupby("speaker_id").size().max() == 3
    assert sa.select_manual_sample(_alignments().assign(status="failed")).empty


def test_manual_token_selection_balances_categories() -> None:
    vowels = {f"u{k}": [{"category": c, "start_s": i * 0.1, "end_s": i * 0.1 + 0.08, "conf": 0.9}
                        for i, c in enumerate(["a"] * 6 + ["ii", "uu", "o"])] for k in range(3)}
    rows = sa.select_manual_tokens(vowels, ["u0", "u1", "u2"])
    assert all(sum(r["utt_id"] == u for r in rows) <= sa.MANUAL_TOKENS_PER_UTT for u in vowels)
    cats = pd.Series([r["category"] for r in rows]).value_counts()
    assert cats["ii"] == 3 and cats["uu"] == 3 and cats["o"] == 3            # rare categories always kept


def _sheet(errors_ms: list[float], present: str = "y", correct: str = "y") -> pd.DataFrame:
    rows = []
    for i, e in enumerate(errors_ms):
        rows.append({"category": "aa" if i % 2 else "u", "auto_start_s": 1.0, "auto_end_s": 1.2, "auto_conf": 0.5 + 0.01 * i,
                     "human_start_s": "=" if e == 0 else 1.0 + e / 1000, "human_end_s": 1.2 + e / 1000,
                     "vowel_present": present, "category_correct": correct})
    return pd.DataFrame(rows)


def test_manual_summary_acceptance() -> None:
    utts = pd.DataFrame({"gross_misalignment": ["n"] * 9 + ["y"]})
    ok = sa.summarise_manual(_sheet([0, 10, 15, 20, 25, 5, 10, 30, 12, 8]), utts)
    assert ok["acceptance"] == "pass" and ok["utterance_failure_rate"] == pytest.approx(0.1)
    assert ok["by_group"]["long"]["n_boundaries"] > 0 and ok["by_group"]["short"]["n_boundaries"] > 0
    bad = sa.summarise_manual(_sheet([60, 70, 80, 90, 20, 30, 100, 120, 5, 10]), utts)
    assert bad["acceptance"] == "fail"
    wrong_cat = sa.summarise_manual(_sheet([5] * 10, correct="n"), utts)
    assert wrong_cat["acceptance"] == "fail" and wrong_cat["category_agreement"] == 0.0
    blank = _sheet([5] * 4).assign(human_start_s="", human_end_s="")
    assert sa.summarise_manual(blank, pd.DataFrame({"gross_misalignment": [""]}))["acceptance"] == "incomplete"


# --------------------------------------------------------------------------- lexical, status, misc


def test_lexical_flag() -> None:
    same = {"a": 50, "aa": 10, "i": 20, "ii": 5, "u": 15}
    assert not sa.lexical_flag(same, same)["flag"]
    skew = {"a": 10, "i": 60, "u": 30}
    res = sa.lexical_flag(same, skew)
    assert res["flag"] and res["max_share_diff"] > sa.LEXICAL_MAX_SHARE_DIFF


def test_measure_status_combines_tests() -> None:
    results = {t: {"aavs_log": {"status": "pass"}} for t in sa.APPLICABLE["aavs_log"]}
    assert sa.measure_status(results, "aavs_log")["overall"] == "pass"
    results["channel"]["aavs_log"] = {"status": "incomplete"}
    assert sa.measure_status(results, "aavs_log")["overall"] == "incomplete"
    results["purity"]["aavs_log"] = {"status": "fail"}
    assert sa.measure_status(results, "aavs_log")["overall"] == "fail"
    assert sa.measure_status({}, "pause_long_share_1s_win")["overall"] == "incomplete"
    assert set(sa.APPLICABLE["pause_long_share_1s_win"]) == {"reliability", "channel"}


def test_monotonic_decrease_and_sufficiency() -> None:
    assert sa.monotonic_decrease([3.0, 2.0, 1.0])
    assert not sa.monotonic_decrease([3.0, 3.0, 1.0]) and not sa.monotonic_decrease([3.0, None, 1.0])
    feats = pd.DataFrame({"speaker_id": ["a"] * 3 + ["b"] * 3, "gate_pass": [True, True, True, True, False, False],
                          "sel_n_selected": [150, 100, 100, 400, 0, 0]})
    suff = sa.speaker_sufficiency(feats).set_index("speaker_id")
    assert suff.loc["a", "formant_sufficient"] and not suff.loc["b", "formant_sufficient"]


def test_synthetic_recovery_small_grid() -> None:
    pytest.importorskip("parselmouth")
    res = sa.synthetic_recovery([(300.0, 2300.0, 120.0), (700.0, 1200.0, 150.0), (450.0, 900.0, 200.0)])
    assert len(res["table"]) == 3 and res["f1_median_rel_err"] < 0.15
    assert set(res) >= {"pass", "f2_median_rel_err"}


def test_envelope_perturbation_requires_pyworld() -> None:
    try:
        import pyworld  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            sa.envelope_perturbation(np.zeros(SR, np.float32), "warp", 0.1)
    else:
        with pytest.raises(ValueError):
            sa.envelope_perturbation(np.zeros(SR, np.float32), "bogus", 0.1)


def test_channel_gate_zero_variance_is_incomplete_not_fail() -> None:
    """Regression: all-zero control measure (e.g. no >1 s pauses in 6 s clips) gave tol=0 → false 'fail'."""
    speakers = {f"r{i}": f"s{i // 2}" for i in range(20)}
    zeros = _features({f"r{i}": 0.0 for i in range(20)}, speakers)
    res = sa.channel_gate(zeros, {"lp": zeros, "hp": _features({f"r{i}": 0.5 for i in range(20)}, speakers)}, "m")
    assert res["status"] == "incomplete" and "variance" in res["reason"]
    assert {c["status"] for c in res["conditions"].values()} == {"not_evaluable"}
