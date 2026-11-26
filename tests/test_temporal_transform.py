from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

pytest.importorskip("parselmouth")

from intelli_speech import acoustic_features as af
from intelli_speech.atypical_profiles import fit_fold, sample_params
from intelli_speech.temporal_transform import LABELS, TemporalParams, transform, wsola
from tests.test_synth_source import _speechlike

SR = 16000


def _feat(x):
    return af.extract_recording_features(x, SR, include_voice=False, include_spectral=False)


def test_wsola_length_pitch_and_determinism() -> None:
    x = _speechlike(0, background=0.003)
    base = _feat(x)
    for factor in (1.3, 1.8):
        y = wsola(x, SR, factor)
        assert abs(y.size / x.size - factor) < 0.01
        f = _feat(y)
        assert abs(f["f0_median_hz"] / base["f0_median_hz"] - 1) < 0.03
        assert abs(f["articulation_rate_nuclei_per_s"] * factor / base["articulation_rate_nuclei_per_s"] - 1) < 0.1
    assert np.array_equal(wsola(x, SR, 1.5), wsola(x, SR, 1.5))


def test_transform_moves_timing_towards_target_and_labels_changes() -> None:
    x = _speechlike(1, background=0.003)
    params = TemporalParams(stretch=1.5, pause_rate_per_min=35, pause_mu=float(np.log(0.9)), pause_sigma=0.4, seed=3)
    y, labels, info = transform(x, SR, params)
    y2, labels2, _ = transform(x, SR, params)
    assert np.array_equal(y, y2) and np.array_equal(labels, labels2)  # reproducible
    before, after = _feat(x), _feat(y)
    assert after["nuclei_count"] == before["nuclei_count"]  # no syllables lost or created
    assert after["pause_rate_per_min_250ms"] > before["pause_rate_per_min_250ms"]
    assert after["pause_median_s_250ms"] > before["pause_median_s_250ms"]
    assert after["speech_activity_ratio"] < before["speech_activity_ratio"]
    assert after["articulation_rate_nuclei_per_s"] < before["articulation_rate_nuclei_per_s"]
    assert abs(after["f0_median_hz"] / before["f0_median_hz"] - 1) < 0.03
    assert abs(labels.size - y.size / (SR * af.FRAME_HOP_S)) <= 2
    assert info["inserted_pauses"] > 0 and info["insertion_shortfall"] == 0
    assert abs(sum(info["label_seconds"].values()) - info["output_s"]) < 0.05
    assert abs(info["speech_out_s"] / info["speech_in_s"] - 1.5) < 0.05


def test_added_pauses_are_background_noise_not_digital_silence() -> None:
    x = _speechlike(2, background=0.003)
    params = TemporalParams(stretch=1.0, pause_rate_per_min=40, pause_mu=float(np.log(0.8)), pause_sigma=0.3, seed=1)
    y, labels, _ = transform(x, SR, params)
    hop = int(SR * af.FRAME_HOP_S)
    n = min(labels.size, y.size // hop)
    frames = y[: n * hop].reshape(n, hop)
    inserted = frames[labels[:n] == LABELS["pause_inserted"]]
    rms = float(np.sqrt(np.mean(inserted.astype(np.float64) ** 2)))
    assert inserted.size and 0.001 < rms < 0.01  # matches the 0.003 background, far below speech


def test_identity_parameters_keep_audio() -> None:
    x = _speechlike(3, background=0.003)
    y, labels, info = transform(x, SR, TemporalParams(stretch=1.0, pause_rate_per_min=0.0, pause_mu=float(np.log(0.26)), pause_sigma=0.1))
    assert info["inserted_pauses"] == 0
    assert abs(y.size - x.size) < int(0.3 * SR)  # only short existing pauses may be lengthened to >= 0.25 s


def _exp1_features() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for s in range(7):
        for r in range(6):
            rows.append({"dataset": "atypical", "group": "atypical", "speaker_id": f"REAL{s}", "record_id": f"a{s}{r}",
                         "articulation_rate_nuclei_per_s": 2.4 + 0.1 * s + 0.05 * rng.normal(),
                         "pause_rate_per_min_250ms": 30 + s + rng.normal(), "pause_median_s_250ms": 0.8 + 0.02 * s,
                         "pause_p90_s_250ms": 1.6 + 0.05 * s})
    for s in range(30):
        for r in range(3):
            rows.append({"dataset": "controls_v2", "group": "typical", "speaker_id": f"T{s}", "record_id": f"t{s}{r}",
                         "articulation_rate_nuclei_per_s": 4.2 + 0.2 * rng.normal(), "pause_rate_per_min_250ms": 15 + rng.normal(),
                         "pause_median_s_250ms": 0.45, "pause_p90_s_250ms": 0.8})
    frame = pd.DataFrame(rows)
    from scripts.validate_temporal_transform import ROBUST_TIMING_FEATURES

    for f in ROBUST_TIMING_FEATURES:
        if f not in frame:
            frame[f] = np.where(frame["group"] == "atypical", 1.0, 0.5) + 0.05 * rng.normal(size=len(frame))
    return frame


def test_fold_excludes_held_out_and_sampling_interpolates() -> None:
    features = _exp1_features()
    alias = {f"REAL{s}": f"S{s + 1}" for s in range(7)}
    fold = fit_fold(features, "REAL3", pseudonyms=alias)
    assert fold["held_out"] == "S4" and len(fold["profiles"]) == 6
    assert "S4" not in {p["source_speaker"] for p in fold["profiles"]}
    assert "REAL" not in json.dumps(fold)  # pseudonyms only
    assert all(p["stretch"] >= 1.3 for p in fold["profiles"])  # 4.2 / (2.4-3.0) = 1.4-1.75
    assert fold["control"]["stretch"] == 1.0
    a = sample_params(fold, np.random.default_rng(5))
    b = sample_params(fold, np.random.default_rng(5))
    assert a == b and 0.3 <= a[1]["alpha"] <= 1.0
    full, _ = sample_params(fold, np.random.default_rng(1), alpha_min=1.0, jitter=1e-9)
    weak, _ = sample_params(fold, np.random.default_rng(1), alpha_min=0.0, jitter=1e-9)
    assert full.stretch >= 1.0 and weak.stretch <= full.stretch + 1e-6


def test_validation_script_end_to_end(tmp_path: Path) -> None:
    from scripts.validate_temporal_transform import main

    features_path = tmp_path / "features.csv"
    _exp1_features().to_csv(features_path, index=False)
    source = tmp_path / "synth_source_v1_full"
    rows, man = [], []
    for c in range(10):
        clip = f"src_{c:04d}"
        path = source / "audio" / f"spk{c % 5}" / f"{clip}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, _speechlike(20 + c, background=0.003), SR, subtype="PCM_16")
        rows.append({"clip_id": clip, "speakerID": f"spk{c % 5}", "usable": True, "split": "train"})
        man.append({"clip_id": clip, "speakerID": f"spk{c % 5}", "file": str(path.relative_to(source))})
    pd.DataFrame(rows).to_csv(source / "quality.csv", index=False)
    pd.DataFrame(man).to_csv(source / "manifest.csv", index=False)
    out = tmp_path / "val"
    args = ["--features", str(features_path), "--source-dir", str(source), "--output-dir", str(out),
            "--clips-per-fold", "8", "--clips-per-speaker", "2", "--examples-per-fold", "1"]
    assert main(args) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["folds"] == 7 and summary["transforms"] == 56 and summary["errors"] == 0
    assert 0.95 < summary["f0_ratio_quantiles"][1] < 1.05
    assert len(list((out / "folds").glob("fold_S*.json"))) == 7
    for path in (out / "folds").glob("*.json"):
        assert "REAL" not in path.read_text()
    assert "REAL" not in (out / "validation_clips.csv").read_text()
    assert main(args) == 1  # refuses to overwrite
