from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.exp1_robustness import compare, run, windows
from tests.test_exp1_pipeline import _features


def test_windows_cover_signal_and_drop_short_tail() -> None:
    x = np.zeros(16000 * 13)
    sizes = [c.size for _, c in windows(x, 6.0)]
    assert sizes == [96000, 96000]  # 1 s tail < half a window is dropped
    sizes = [c.size for _, c in windows(np.zeros(16000 * 15), 6.0)]
    assert sizes == [96000, 96000, 48000]


def test_compare_requires_class1_everywhere() -> None:
    a = pd.DataFrame({"feature": ["x", "y"], "family": ["t", "t"], "verdict": ["candidate_class1"] * 2,
                      "consistency_k": [7, 7], "cliffs_delta": [1.0, 1.0]})
    b = a.assign(verdict=["candidate_class1", "not_supported"])
    out = compare({"main": a, "other": b}).set_index("feature")
    assert out.loc["x", "robust_class1"] and not out.loc["y", "robust_class1"]


def test_run_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("parselmouth")
    from tests.test_acoustic_features import synthetic_speech

    feats = _features()
    copy = feats[feats.dataset == "controls_v2"].assign(dataset="vaani_copy")
    feats = pd.concat([feats, copy], ignore_index=True).assign(duration_s=6.0)
    path = tmp_path / "features.csv"
    feats.to_csv(path, index=False)
    records = ({"speaker_id": f"A{s}", "record_id": f"atyp_{s}", "rate": 16000,
                "audio": synthetic_speech(pause_s=0.8, phrases=6, seed=s)[:, None]} for s in range(7))
    args = argparse.Namespace(features=path, hf_cache=None, output_dir=None, window_s=None, permutations=100,
                              skip_duration_matched=False, overwrite=False)
    info = run(args, records=records)
    assert set(info["checks"]) == {"main", "vs_vaani_copy", "duration_matched"}
    assert info["duration_matched"]["windows_ok"] >= 7
    out = tmp_path / "robustness"
    assert (out / "robustness_summary.csv").is_file()
    saved = json.loads((out / "robustness.json").read_text())
    assert saved["duration_matched"]["window_s"] == 6.0
    with pytest.raises(FileExistsError):
        run(args, records=iter(()))
