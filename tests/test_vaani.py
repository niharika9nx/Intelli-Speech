from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from intelli_speech.vaani import (
    decode_audio_bytes,
    iter_atypical,
    iter_typical_vaani,
    load_typical_vaani_metadata,
    pcm_sha1,
)
from scripts.audit_v2 import atypical_extras, audit_records, cramers_v, main, speaker_table

SR = 16000


def _wav_bytes(seconds: float = 0.5, freq: float = 220.0, channels: int = 1) -> bytes:
    t = np.arange(int(SR * seconds)) / SR
    tone = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    buffer = io.BytesIO()
    sf.write(buffer, data, SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _fake_atypical_rows() -> list[dict]:
    rows = []
    for i, (speaker, level) in enumerate([("s1", "Low"), ("s1", "Low"), ("s2", "High"), ("s2", "High")]):
        rows.append(
            {
                "audio": {"bytes": _wav_bytes(0.5 + 0.1 * i, 200 + 50 * i), "path": f"clip{i}.wav"},
                "speaker_id": speaker,
                "language": "Telugu",
                "prompt_file_type": "image",
                "prompt_file_name": f"img{i % 2}.jpg",
                "transcription": "తెలుగు భాష",
                "difficulty_level": level,
                "verbatim_transcription": "తెలుగు" if i else "",
                "duration_s": 0.5 + 0.1 * i,
            }
        )
    rows.append({**rows[0], "audio": {"bytes": b"not audio", "path": "bad.wav"}})
    return rows


def _make_typical_dir(root: Path) -> Path:
    folder = root / "vaani_telugu_transcribed"
    folder.mkdir(parents=True)
    names = []
    for i in range(3):
        name = f"telugu_{i:04d}.wav"
        (folder / name).write_bytes(_wav_bytes(1.0, 180 + 40 * i))
        names.append(name)
    frame = pd.DataFrame(
        {
            "filename": names,
            "duration": ["1.0"] * 3,
            "language": ["Telugu"] * 3,
            "speakerID": ["a", "a", "b"],
            "state": ["AP"] * 3,
            "district": ["d1", "d2", "d1"],
            "pincode": ["1"] * 3,
            "transcript": ["తెలుగు", "", "భాష"],
            "source_audio_path": [""] * 3,
            "audio_bytes": [""] * 3,
        }
    )
    frame.to_csv(folder / "metadata.csv", index=False, encoding="utf-8-sig")
    return folder


def _make_slr66_dir(root: Path) -> Path:
    folder = root / "openslr_slr66"
    folder.mkdir(parents=True)
    for gender, prefix in (("female", "tef"), ("male", "tem")):
        ids = [f"{prefix}_0001_00000{i}" for i in range(3)]
        (folder / f"line_index_{gender}.tsv").write_text(
            "\n".join(f"{i}\tతెలుగు" for i in ids), encoding="utf-8"
        )
        with zipfile.ZipFile(folder / f"te_in_{gender}.zip", "w") as archive:
            for i in ids:
                archive.writestr(f"{i}.wav", _wav_bytes(0.3))
    return folder


def test_decode_and_hash_detects_duplicates() -> None:
    a, rate, channels = decode_audio_bytes(_wav_bytes(channels=2))
    assert rate == SR and channels == 2 and a.shape[1] == 2
    assert pcm_sha1(a) == pcm_sha1(decode_audio_bytes(_wav_bytes(channels=2))[0])


def test_iter_atypical_reports_decode_errors_without_stopping() -> None:
    records = list(iter_atypical(_fake_atypical_rows()))
    assert len(records) == 5
    assert "decode_error" in records[-1].metadata
    assert all("decode_error" not in r.metadata for r in records[:4])


def test_atypical_audit_tables_and_confound_measure() -> None:
    rows, summary = audit_records(iter_atypical(_fake_atypical_rows()), kind="atypical")
    extras = atypical_extras(rows)
    assert summary["files"] == 5 and summary["decoded"] == 4 and summary["speakers"] == 2
    # difficulty is perfectly determined by speaker in the fixture -> V == 1
    assert extras["difficulty_speaker_cramers_v"] == pytest.approx(1.0)
    assert extras["verbatim_empty"] == 1
    assert extras["prompts_shared_by_2plus_speakers"] == 2
    assert all("transcription" not in r and "verbatim_transcription" not in r for r in rows)
    table = speaker_table(rows, kind="atypical")
    assert {t["speaker_id"] for t in table} == {"s1", "s2"}


def test_cramers_v_degenerate_returns_none() -> None:
    from collections import Counter

    assert cramers_v({"s1": Counter({"Low": 3})}) is None


def test_typical_loader_handles_bom(tmp_path: Path) -> None:
    folder = _make_typical_dir(tmp_path)
    frame = load_typical_vaani_metadata(folder)
    assert "filename" in frame.columns
    records = list(iter_typical_vaani(folder, frame))
    assert len(records) == 3 and {r.speaker_id for r in records} == {"a", "b"}


def test_audit_cli_typical_and_slr66(tmp_path: Path) -> None:
    _make_typical_dir(tmp_path)
    _make_slr66_dir(tmp_path)
    out = tmp_path / "analysis_v2" / "audit_test"
    code = main(["--data-root", str(tmp_path), "--output-dir", str(out), "--skip-atypical", "--slr66-sample", "2"])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["typical_vaani"]["speakers"] == 2
    assert summary["typical_vaani"]["wav_files_on_disk"] == 3
    assert summary["slr66"]["female"]["index_rows"] == 3
    assert summary["slr66"]["male"]["decode_sample"]["files"] == 2
    assert (out / "typical_per_speaker.csv").is_file()
    # second run refuses to overwrite
    assert main(["--data-root", str(tmp_path), "--output-dir", str(out), "--skip-atypical"]) == 1


def test_audit_refuses_output_inside_dataset(tmp_path: Path) -> None:
    folder = _make_typical_dir(tmp_path)
    assert main(["--data-root", str(tmp_path), "--output-dir", str(folder / "x"), "--skip-atypical", "--skip-slr66"]) == 1
