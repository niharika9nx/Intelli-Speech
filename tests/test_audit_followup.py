from __future__ import annotations

import numpy as np
import pytest

from intelli_speech.audio_integrity import full_scale_analysis
from intelli_speech.grouped_stats import cliffs_delta, speaker_equal_weight_mean, stratified_permutation_test
from intelli_speech.text_audit import script_profile
from scripts.audit_v2_followup import difficulty_analysis, integrity_summary, pseudonyms

SR = 16000


def _speechlike(seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    x = np.sin(2 * np.pi * 150 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.05 * rng.standard_normal(t.size)
    return x


def _pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)


def test_peak_normalised_signal_is_not_called_clipping() -> None:
    x = _speechlike()
    x = x / np.abs(x).max() * (32767 / 32768)
    result = full_scale_analysis(_pcm16(x), is_pcm16=True)
    assert result["verdict"] == "consistent_with_peak_normalisation"
    assert result["max_consecutive_near_full_scale"] <= 2


def test_hard_clipped_signal_shows_runs() -> None:
    x = np.clip(_speechlike() * 3.0, -1.0, 1.0)
    result = full_scale_analysis(_pcm16(x), is_pcm16=True)
    assert result["verdict"] == "clipping_evidence"
    assert result["max_consecutive_near_full_scale"] >= 3
    assert result["exact_full_scale_samples"] > 10


def test_quiet_signal_and_float_path() -> None:
    result = full_scale_analysis((_speechlike() * 0.3).astype(np.float32), is_pcm16=False)
    assert result["verdict"] == "below_full_scale"
    with pytest.raises(ValueError):
        full_scale_analysis(np.zeros(0), is_pcm16=False)


def test_script_profile_proportions_and_missing() -> None:
    profile = script_profile("తెలుగు abc [noise] 12")
    assert profile["prop_telugu"] > 0 and profile["prop_latin"] > 0
    assert profile["prop_digit"] > 0 and profile["prop_symbol"] > 0
    assert set(profile["symbols"]) == {"[", "]"}
    assert profile["aksharas"] == 3
    assert script_profile(None)["missing"] and script_profile("  ")["missing"]
    assert script_profile("hello world")["dominant_class"] == "latin"


def test_within_speaker_test_ignores_between_speaker_differences() -> None:
    rng = np.random.default_rng(1)
    values, labels, groups = [], [], []
    # Speaker A mostly Medium with high values, speaker B mostly Low with low values,
    # but NO within-speaker Medium/Low difference.
    for speaker, level_mix, offset in (("A", ["Medium"] * 8 + ["Low"] * 4, 10.0), ("B", ["Low"] * 8 + ["Medium"] * 4, 0.0)):
        for level in level_mix:
            values.append(offset + rng.normal())
            labels.append(level)
            groups.append(speaker)
    pooled = cliffs_delta([v for v, l in zip(values, labels) if l == "Medium"], [v for v, l in zip(values, labels) if l == "Low"])
    within = stratified_permutation_test(values, labels, groups, positive="Medium", negative="Low", n_permutations=500)
    assert pooled > 0.2  # spurious pooled association driven by speaker
    assert within["eligible_groups"] == 2
    assert within["p_value"] > 0.05


def test_within_speaker_test_detects_real_effect() -> None:
    rng = np.random.default_rng(2)
    values, labels, groups = [], [], []
    for speaker in ("A", "B", "C"):
        for level in ["Medium"] * 6 + ["Low"] * 6:
            values.append((2.0 if level == "Medium" else 0.0) + 0.3 * rng.normal())
            labels.append(level)
            groups.append(speaker)
    within = stratified_permutation_test(values, labels, groups, positive="Medium", negative="Low", n_permutations=500)
    assert within["p_value"] < 0.05 and within["sign_consistency"] == 1.0


def test_speaker_equal_weight_mean() -> None:
    assert speaker_equal_weight_mean([1, 1, 1, 1, 10], ["a", "a", "a", "a", "b"]) == pytest.approx(5.5)


def test_difficulty_analysis_keeps_direction_unknown() -> None:
    rows = []
    for i in range(30):
        rows.append({"speaker_id": "AB"[i % 2], "difficulty_level": "Medium" if i % 3 else "Low",
                     "cer_verbatim_vs_intended": float(i % 5), "duration_s": 30.0 + i})
    rows.append({"speaker_id": "A", "difficulty_level": "High", "duration_s": 31.0})
    result = difficulty_analysis(rows)
    assert result["direction_conclusion"].startswith("UNKNOWN")
    assert "cer_verbatim_vs_intended" in result["features"]
    assert result["levels"]["High"] == 1


def test_integrity_summary_and_pseudonyms() -> None:
    rows = [{"speaker_id": "x", "verdict": "clipping_evidence", "exact_full_scale_samples": 5, "near_full_scale_samples": 9,
             "near_full_scale_fraction": 0.01, "max_consecutive_near_full_scale": 4, "subtype": "PCM_16"},
            {"speaker_id": "y", "decode_error": "bad"}]
    summary = integrity_summary(rows)
    assert summary["analysed"] == 1 and summary["files_with_run_ge_3"] == 1
    assert pseudonyms(["b", "a", "b"]) == {"a": "S1", "b": "S2"}
