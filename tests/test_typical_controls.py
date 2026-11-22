from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
fsspec = pytest.importorskip("fsspec")
from fsspec.implementations.local import LocalFileSystem

from intelli_speech.vaani_controls import (
    SelectionConfig,
    estimate_audio_bytes,
    fetch_selected,
    list_telugu_shards,
    scan_metadata,
    select_controls,
    spread_subset,
)


def _wav(seconds: float, rate: int = 48000, freq: float = 200.0) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    buffer = io.BytesIO()
    sf.write(buffer, (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _metadata(n_speakers: int, clips: int, gender_cycle=("Female", "Male")) -> pd.DataFrame:
    rows = []
    for s in range(n_speakers):
        for u in range(clips):
            rows.append({"language": "Telugu", "duration": 3.0 + u % 4, "speakerID": f"spk{s:03d}",
                         "gender": gender_cycle[s % len(gender_cycle)], "district": f"d{s % 5}",
                         "isTranscriptionAvailable": "True", "transcript": "తెలుగు", "UtteranceSequenceID": u,
                         "_shard": "s0", "_row_group": 0, "_row": len(rows)})
    return pd.DataFrame(rows)


def test_selection_respects_caps_and_is_deterministic() -> None:
    meta = _metadata(60, 15)
    a, report = select_controls(meta)
    b, _ = select_controls(meta.sample(frac=1.0, random_state=3))
    assert report["selected_speakers"] == 40 and len(a) == 400
    assert a.groupby("speakerID").size().max() <= 10
    assert sorted(zip(a.speakerID, a._utt)) == sorted(zip(b.speakerID, b._utt))
    genders = a.drop_duplicates("speakerID")["gender"].value_counts()
    assert abs(genders["Female"] - genders["Male"]) <= 1


def test_selection_filters_and_reports_shortfall() -> None:
    meta = _metadata(10, 4)
    meta.loc[meta.index[:4], "language"] = "Hindi"
    meta.loc[meta.index[4:8], "transcript"] = ""
    meta.loc[meta.index[8], "duration"] = 99.0
    selected, report = select_controls(meta, SelectionConfig(n_speakers=40))
    assert report["shortfall"] is True
    assert report["selected_speakers"] < 40
    assert (selected["language"] == "Telugu").all()
    # duplicates of (speaker, utterance) are dropped
    doubled = pd.concat([_metadata(5, 5), _metadata(5, 5)])
    sel, rep = select_controls(doubled, SelectionConfig(n_speakers=5))
    assert len(sel) == 25


def test_spread_subset_even_and_deterministic() -> None:
    items = [f"x{i:02d}" for i in range(10)]
    assert spread_subset(items, 3) == ["x00", "x04", "x09"] or spread_subset(items, 3)[0] == "x00"
    assert spread_subset(items, None) == items


class PrefixFS(LocalFileSystem):
    """Maps 'datasets/...' paths onto a local root (stands in for HfFileSystem)."""

    def __init__(self, root: Path):
        super().__init__()
        self._root = Path(root)

    def _map(self, path: str) -> str:
        return str(self._root / path) if str(path).startswith("datasets/") else path

    def glob(self, path, **kwargs):
        return [str(Path(p).relative_to(self._root)).replace("\\", "/") for p in super().glob(self._map(path), **kwargs)]

    def open(self, path, mode="rb", **kwargs):
        return super().open(self._map(path), mode, **kwargs)


def _write_shards(root: Path, n_shards: int = 2) -> None:
    folder = root / "datasets" / "ARTPARK-IISc" / "Vaani" / "audio" / "Telugu"
    folder.mkdir(parents=True)
    for shard in range(n_shards):
        meta = _metadata(6, 4)
        meta["speakerID"] = meta["speakerID"] + f"_{shard}"
        audio = [{"bytes": _wav(0.5 + 0.1 * i, freq=150 + 10 * i + shard), "path": f"{i}.wav"} for i in range(len(meta))]
        table = pa.Table.from_pandas(meta.drop(columns=["_shard", "_row_group", "_row"]), preserve_index=False)
        table = table.append_column("audio", pa.array(audio, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])))
        pq.write_table(table, folder / f"train-{shard:05d}.parquet", row_group_size=10)


def test_scan_select_fetch_end_to_end(tmp_path: Path) -> None:
    _write_shards(tmp_path)
    fs = PrefixFS(tmp_path)
    shards = list_telugu_shards(fs)
    assert len(shards) == 2
    meta = scan_metadata(fs, shards)
    assert len(meta) == 48 and "audio" not in meta.columns
    selected, report = select_controls(meta, SelectionConfig(n_speakers=5, max_per_speaker=3, max_total=12))
    assert report["selected_speakers"] == 4 and len(selected) == 12  # max_total stops at 12
    assert estimate_audio_bytes(fs, selected) > 0
    out = tmp_path / "typical_controls_v2"
    manifest = fetch_selected(fs, selected, out, source_revision="abc123")
    ok = manifest[manifest.status == "ok"]
    assert len(ok) == 12 and set(ok.sample_rate) == {16000} and ok.resampled.all()
    assert set(ok.original_sample_rate) == {48000} and ok.pcm_sha1.is_unique
    first = out / ok.iloc[0]["file"]
    data, rate = sf.read(first)
    assert rate == 16000 and data.ndim == 1


def test_build_script_plan_then_download(tmp_path: Path, monkeypatch) -> None:
    import scripts.build_typical_controls as build

    _write_shards(tmp_path)
    monkeypatch.setattr(build, "source_revision", lambda: "rev1")
    fs = PrefixFS(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()
    base = ["--data-root", str(data_root), "--n-speakers", "4", "--max-per-speaker", "3", "--max-total", "12"]
    plan = build.run(build.build_parser().parse_args(["--phase", "plan", *base]), fs=fs)
    assert plan["selected_rows"] == 12 and plan["source_revision"] == "rev1"
    stats = build.run(build.build_parser().parse_args(["--phase", "download", *base]), fs=fs)
    assert stats["files_written"] == 12 and stats["speakers"] == 4
    out = data_root / "typical_controls_v2"
    assert (out / "manifest.csv").is_file() and json.loads((out / "provenance.json").read_text())["license"].startswith("CC-BY-4.0")
    with pytest.raises(FileExistsError):
        build.run(build.build_parser().parse_args(["--phase", "download", *base]), fs=fs)
    with pytest.raises(FileExistsError):
        build.run(build.build_parser().parse_args(["--phase", "plan", *base]), fs=fs)


def test_placeholder_ids_filtered_and_missing_utterance_ids_not_collapsed() -> None:
    meta = _metadata(8, 6)
    meta.loc[meta.speakerID == "spk000", "speakerID"] = "None"  # placeholder, must not become a speaker
    meta.loc[meta.speakerID == "spk001", "UtteranceSequenceID"] = None  # must not collapse to 1 row
    selected, report = select_controls(meta, SelectionConfig(n_speakers=10, max_per_speaker=6, max_total=100,
                                                             min_per_speaker=5, require_transcript=False))
    assert report["invalid_speaker_id_rows"] == 6 and report["missing_utterance_id_rows"] == 6
    assert "None" not in set(selected.speakerID)
    assert (selected.speakerID == "spk001").sum() == 6 and report["selected_speakers"] == 7


def test_block_subset_takes_consecutive_runs() -> None:
    from intelli_speech.vaani_controls import block_subset

    items = [f"s{i:03d}" for i in range(100)]
    chosen = block_subset(items, 9, 3)
    idx = [items.index(c) for c in chosen]
    assert len(chosen) == 9 and idx[:3] == [0, 1, 2] and idx[-1] == 99
    assert all(idx[i + 1] - idx[i] == 1 for i in (0, 1, 3, 4, 6, 7))
    assert block_subset(items, None, 3) == items
