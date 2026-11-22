from __future__ import annotations

import csv
from pathlib import Path

import pytest

from intelli_speech.dataset import (
    filter_split,
    iter_pairs,
    load_metadata,
    load_pairs,
    split_counts,
)

COLUMNS = [
    "source_id",
    "split",
    "clean_file",
    "noisy_file",
    "target_snr_db",
    "sample_rate",
    "duration_sec",
    "noise_environment",
    "noise_file",
    "gender",
    "speaker_id",
    "transcript",
]


def make_dataset(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    (tmp_path / "clean").mkdir()
    (tmp_path / "noisy").mkdir()
    for row in rows:
        (tmp_path / str(row["clean_file"])).touch()
        (tmp_path / str(row["noisy_file"])).touch()
    with (tmp_path / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def rows() -> list[dict[str, object]]:
    return [
        {
            "source_id": "s1",
            "split": "train",
            "clean_file": "clean/s1.wav",
            "noisy_file": "noisy/s1_snr5.wav",
            "target_snr_db": 5,
            "sample_rate": 16000,
            "duration_sec": 1.0,
            "noise_environment": "DKITCHEN",
            "noise_file": "DKITCHEN/a.wav",
            "gender": "F",
            "speaker_id": "spk1",
            "transcript": "hello",
        },
        {
            "source_id": "s1",
            "split": "train",
            "clean_file": "clean/s1.wav",
            "noisy_file": "noisy/s1_snr10.wav",
            "target_snr_db": 10,
            "sample_rate": 16000,
            "duration_sec": 1.0,
            "noise_environment": "NPARK",
            "noise_file": "NPARK/b.wav",
            "gender": "F",
            "speaker_id": "spk1",
            "transcript": "hello",
        },
        {
            "source_id": "s2",
            "split": "test",
            "clean_file": "clean/s2.wav",
            "noisy_file": "noisy/s2_snr5.wav",
            "target_snr_db": 5,
            "sample_rate": 16000,
            "duration_sec": 2.0,
            "noise_environment": "OOFFICE",
            "noise_file": "OOFFICE/c.wav",
            "gender": "M",
            "speaker_id": "spk2",
            "transcript": "world",
        },
        {
            "source_id": "s3",
            "split": "validation",
            "clean_file": "clean/s3.wav",
            "noisy_file": "noisy/s3_snr10.wav",
            "target_snr_db": 10,
            "sample_rate": 16000,
            "duration_sec": 1.5,
            "noise_environment": "TBUS",
            "noise_file": "TBUS/d.wav",
            "gender": "F",
            "speaker_id": "spk3",
            "transcript": "validation",
        },
    ]


def test_load_metadata_and_split_counts(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, rows())

    metadata = load_metadata(dataset)

    assert len(metadata) == 4
    assert split_counts(metadata) == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }


def test_filter_split_returns_reset_copy(tmp_path: Path) -> None:
    metadata = load_metadata(make_dataset(tmp_path, rows()))

    test = filter_split(metadata, "test")

    assert list(test["source_id"]) == ["s2"]
    assert list(test.index) == [0]
    assert len(metadata) == 4


def test_iter_pairs_resolves_paths_and_fields(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, rows())

    pairs = load_pairs(dataset, split="test")

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.source_id == "s2"
    assert pair.clean_path == dataset / "clean" / "s2.wav"
    assert pair.noisy_path == dataset / "noisy" / "s2_snr5.wav"
    assert pair.target_snr_db == 5.0
    assert pair.sample_rate == 16000
    assert pair.noise_environment == "OOFFICE"
    assert pair.transcript == "world"


def test_iter_pairs_limit(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, rows())

    pairs = list(iter_pairs(dataset, split="train", limit=1))

    assert len(pairs) == 1
    assert pairs[0].target_snr_db == 5.0


def test_missing_required_column(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, rows())
    text = (dataset / "metadata.csv").read_text(encoding="utf-8")
    text = text.replace("clean_file", "wrong_column", 1)
    (dataset / "metadata.csv").write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_metadata(dataset)


def test_invalid_split_value(tmp_path: Path) -> None:
    bad_rows = rows()
    bad_rows[0]["split"] = "dev"
    dataset = make_dataset(tmp_path, bad_rows)

    with pytest.raises(ValueError, match="invalid splits"):
        load_metadata(dataset)


def test_missing_audio_file_is_reported(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, rows())
    (dataset / "clean" / "s2.wav").unlink()

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        load_pairs(dataset, split="test")


def test_duplicate_noisy_file_is_rejected(tmp_path: Path) -> None:
    bad_rows = rows()
    bad_rows[1]["noisy_file"] = bad_rows[0]["noisy_file"]
    dataset = make_dataset(tmp_path, bad_rows)

    with pytest.raises(ValueError, match="noisy_file"):
        load_metadata(dataset)


def test_missing_dataset_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
        load_metadata(tmp_path / "missing")


def test_empty_split_is_rejected(tmp_path: Path) -> None:
    metadata = load_metadata(make_dataset(tmp_path, rows()))
    empty_metadata = metadata.iloc[0:0]

    with pytest.raises(ValueError, match="contains no metadata rows"):
        filter_split(empty_metadata, "test")
