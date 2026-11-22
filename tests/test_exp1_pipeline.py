from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_exp1 import analyse, bh_fdr, icc1, main as analyse_main


def _features(seed: int = 0) -> pd.DataFrame:
    """7 atypical speakers with consistently longer pauses; 30 control speakers.

    'pitch_like' differs for only one atypical speaker; 'channel_driven' tracks bandwidth.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(7):
        n = 5 if s < 2 else 30
        bw = rng.uniform(1500, 4000)
        for r in range(n):
            rows.append({"dataset": "atypical", "group": "atypical", "speaker_id": f"A{s}", "record_id": f"a{s}_{r}",
                         "pause_median_s_250ms": 0.9 + 0.1 * rng.normal(),
                         "f0_sd_st": (6.0 if s == 0 else 2.0) + 0.3 * rng.normal(),
                         "spectral_centroid_hz_median": bw * 0.4 + 20 * rng.normal(),
                         "effective_bandwidth_hz": bw + 50 * rng.normal(), "rms_dbfs": -20 + rng.normal()})
    for s in range(30):
        bw = rng.uniform(1500, 4000)
        for r in range(6):
            rows.append({"dataset": "controls_v2", "group": "typical", "speaker_id": f"T{s}", "record_id": f"t{s}_{r}",
                         "pause_median_s_250ms": 0.45 + 0.1 * rng.normal(),
                         "f0_sd_st": 2.0 + 0.3 * rng.normal(),
                         "spectral_centroid_hz_median": bw * 0.4 + 20 * rng.normal(),
                         "effective_bandwidth_hz": bw + 50 * rng.normal(), "rms_dbfs": -20 + rng.normal()})
    return pd.DataFrame(rows)


def test_catalogue_verdicts_follow_criteria() -> None:
    catalogue, speakers = analyse(_features(), control_dataset="controls_v2", n_perm=500, n_boot=200)
    verdict = dict(zip(catalogue.feature, catalogue.verdict))
    assert verdict["pause_median_s_250ms"] == "candidate_class1"
    assert verdict["f0_sd_st"] == "speaker_specific"
    assert verdict["spectral_centroid_hz_median"] == "channel_confounded"
    row = catalogue.set_index("feature").loc["pause_median_s_250ms"]
    assert row["consistency_k"] == 7 and row["loso_sign_stable"] and row["direction"] == "higher"
    assert set(speakers.speaker_id) >= {"A0", "T0"}


def test_icc_and_fdr_helpers() -> None:
    values = pd.Series([1, 1.1, 0.9, 5, 5.1, 4.9])
    groups = pd.Series(["a", "a", "a", "b", "b", "b"])
    assert icc1(values, groups) > 0.9
    q = bh_fdr([0.01, 0.04, None, 0.5])
    assert q[2] is None and q[0] <= q[1] <= q[3]


def test_insufficient_data_is_reported() -> None:
    small = _features()
    small = small[small.speaker_id.isin(["A0", "A1", "T0", "T1"])]
    catalogue, _ = analyse(small, control_dataset="controls_v2", n_perm=100, n_boot=50)
    assert set(catalogue.verdict) == {"insufficient_data"}


def test_cli_writes_outputs(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _features().to_csv(path, index=False)
    assert analyse_main(["--features", str(path), "--permutations", "200"]) == 0
    summary = json.loads((tmp_path / "analysis" / "analysis_summary.json").read_text())
    assert summary["control_dataset"] == "controls_v2"
    assert (tmp_path / "analysis" / "catalogue.csv").is_file()
    assert analyse_main(["--features", str(path), "--permutations", "50"]) == 1  # refuses to overwrite


def test_extract_script_on_fixture_sources(tmp_path: Path) -> None:
    pytest.importorskip("parselmouth")
    from scripts.extract_features_exp1 import main as extract_main
    from tests.test_acoustic_features import synthetic_speech
    import soundfile as sf

    root = tmp_path / "data"
    folder = root / "vaani_telugu_transcribed"
    folder.mkdir(parents=True)
    names = []
    for i in range(3):
        name = f"telugu_{i:04d}.wav"
        sf.write(folder / name, synthetic_speech(pause_s=0.3 + 0.1 * i, phrases=3, seed=i), 16000, subtype="PCM_16")
        names.append(name)
    pd.DataFrame({"filename": names, "speakerID": ["a", "a", "b"], "transcript": ["x"] * 3, "district": ["d"] * 3}).to_csv(
        folder / "metadata.csv", index=False, encoding="utf-8-sig")
    out = root / "analysis_v2" / "exp1_test"
    assert extract_main(["--data-root", str(root), "--output-dir", str(out), "--sources", "vaani_copy", "controls_v2", "--no-voice"]) == 0
    features = pd.read_csv(out / "features.csv")
    assert len(features) == 3 and "pause_median_s_250ms" in features
    assert "error" not in features or features["error"].isna().all()
    config = json.loads((out / "run_config.json").read_text())
    assert config["counts"]["controls_v2"]["skipped"].startswith("missing")
