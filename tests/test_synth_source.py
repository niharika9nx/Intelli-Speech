from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

pytest.importorskip("pyarrow")
pytest.importorskip("fsspec")

from intelli_speech.source_quality import QualityConfig, assign_speaker_splits, clip_quality, failing_reasons
from tests.test_acoustic_features import synthetic_speech
from tests.test_typical_controls import PrefixFS, _write_shards

SR = 16000


def _speechlike(seed: int, background: float = 0.0) -> np.ndarray:
    """Harmonic syllables with broadband 'frication' inside bursts, optional constant noise."""
    rng = np.random.default_rng(seed)
    x = synthetic_speech(phrases=4, seed=seed)
    envelope = np.convolve(np.abs(x) > 0.01, np.ones(160) / 160, mode="same")
    x = x + 0.05 * rng.standard_normal(x.size) * envelope
    if background:
        x = x + background * rng.standard_normal(x.size)
    return np.clip(x, -1, 1).astype(np.float32)


def test_quality_filter_accepts_clean_and_rejects_noisy_or_short() -> None:
    config = QualityConfig()
    clean = clip_quality(_speechlike(0), SR)
    assert failing_reasons(clean, config) == [], clean
    noisy = clip_quality(_speechlike(1, background=0.1), SR)
    assert "low_snr" in failing_reasons(noisy, config)
    short = clip_quality(_speechlike(2)[: int(0.9 * SR)], SR)
    assert "little_speech" in failing_reasons(short, config)


def test_speaker_splits_deterministic_and_stratified() -> None:
    speakers = pd.DataFrame({"speakerID": [f"s{i}" for i in range(40)], "gender": ["Female", "Male"] * 20})
    a = assign_speaker_splits(speakers, salt="x")
    b = assign_speaker_splits(speakers.sample(frac=1.0, random_state=1).reset_index(drop=True), salt="x")
    assert dict(zip(speakers.speakerID, a)) == dict(zip(speakers.sample(frac=1.0, random_state=1).speakerID, b))
    counts = pd.crosstab(speakers["gender"], a)
    assert (counts["train"] == 16).all() and (counts["val"] == 2).all() and (counts["test"] == 2).all()


def _controls(data_root: Path) -> None:
    (data_root / "typical_controls_v2").mkdir(parents=True)
    pd.DataFrame({"speakerID": ["spk000_0"], "status": ["ok"]}).to_csv(data_root / "typical_controls_v2" / "manifest.csv", index=False)
    (data_root / "vaani_telugu_transcribed").mkdir()
    pd.DataFrame({"filename": ["a.wav"], "speakerID": ["spk001_1"], "transcript": ["x"]}).to_csv(
        data_root / "vaani_telugu_transcribed" / "metadata.csv", index=False, encoding="utf-8-sig")


def test_plan_excludes_control_speakers_and_download_writes_src_ids(tmp_path: Path, monkeypatch) -> None:
    import scripts.build_synth_source as build

    _write_shards(tmp_path)
    monkeypatch.setattr(build, "source_revision", lambda: "rev1")
    fs = PrefixFS(tmp_path)
    data_root = tmp_path / "data"
    base = ["--data-root", str(data_root), "--n-speakers", "20", "--max-per-speaker", "3", "--max-total", "100",
            "--min-per-speaker", "3", "--max-shards", "2"]
    data_root.mkdir()
    with pytest.raises(FileNotFoundError):  # control lists missing → refuse
        build.run(build.build_parser().parse_args(["--phase", "plan", *base]), fs=fs)
    _controls(data_root)
    plan = build.run(build.build_parser().parse_args(["--phase", "plan", *base]), fs=fs)
    assert plan["rows_removed_by_exclusion"] == 8 and plan["exclusions"]["total_excluded_speakers"] == 2
    out = data_root / "synth_source_v1_pilot"
    selected = pd.read_csv(out / "selection_plan.csv", dtype={"speakerID": str})
    assert not selected["speakerID"].isin(["spk000_0", "spk001_1"]).any()
    assert selected.groupby("speakerID").size().max() <= 3 and plan["selected_speakers"] == 10
    stats = build.run(build.build_parser().parse_args(["--phase", "download", *base]), fs=fs)
    assert stats["files_written"] == 30
    manifest = pd.read_csv(out / "manifest.csv")
    assert manifest["clip_id"].str.startswith("src_").all()
    with pytest.raises(FileExistsError):
        build.run(build.build_parser().parse_args(["--phase", "download", *base]), fs=fs)


def test_quality_phase_marks_and_splits(tmp_path: Path) -> None:
    pytest.importorskip("parselmouth")
    import scripts.build_synth_source as build

    data_root = tmp_path / "data"
    out = data_root / "synth_source_v1_pilot"
    rows = []
    for s in range(6):
        for c in range(6):
            noisy = s == 5 or c == 5  # speaker 5 all noisy; one noisy clip per speaker
            clip = f"src_{len(rows):04d}"
            path = out / "audio" / f"spk{s}" / f"{clip}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, _speechlike(10 * s + c, background=0.1 if noisy else 0.0), SR, subtype="PCM_16")
            rows.append({"clip_id": clip, "speakerID": f"spk{s}", "gender": ["Female", "Male"][s % 2], "district": "d",
                         "status": "ok", "file": str(path.relative_to(out)), "duration_s": 5.0, "has_transcript": c == 0})
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    summary = build.run(build.build_parser().parse_args(["--phase", "quality", "--data-root", str(data_root)]))
    assert summary["clips_screened"] == 36 and summary["speakers_kept"] == 5
    assert summary["usable_clips"] == 25 and summary["reject_reasons"]["low_snr"] >= 6
    quality = pd.read_csv(out / "quality.csv")
    assert quality.loc[~quality.quality_pass, "split"].isna().all()
    assert set(quality.loc[quality.usable, "split"]) <= {"train", "val", "test"}
    assert json.loads((out / "quality_summary.json").read_text())["thresholds"]["min_snr_db"] == 15.0
    with pytest.raises(FileExistsError):
        build.run(build.build_parser().parse_args(["--phase", "quality", "--data-root", str(data_root)]))
