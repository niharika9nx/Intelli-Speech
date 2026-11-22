from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.evaluate_baselines import evaluate, main, prepare_output_dir

COLUMNS = [
    "source_id",
    "split",
    "clean_file",
    "noisy_file",
    "target_snr_db",
]


def make_audio_dataset(tmp_path: Path) -> Path:
    (tmp_path / "clean").mkdir(parents=True)
    (tmp_path / "noisy").mkdir()
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    clean = (0.25 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    rng = np.random.default_rng(17)
    noise = rng.normal(size=clean.shape)
    noise *= np.sqrt(np.mean(clean**2)) / (
        np.sqrt(np.mean(noise**2)) * 10 ** (5 / 20)
    )
    noisy = (clean + noise).astype(np.float32)
    sf.write(tmp_path / "clean" / "s1.wav", clean, sample_rate, subtype="PCM_16")
    sf.write(tmp_path / "noisy" / "s1_snr5.wav", noisy, sample_rate, subtype="PCM_16")
    with (tmp_path / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "source_id": "s1",
                "split": "test",
                "clean_file": "clean/s1.wav",
                "noisy_file": "noisy/s1_snr5.wav",
                "target_snr_db": 5,
            }
        )
    return tmp_path


def test_prepare_output_dir_refuses_nonempty(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_output_dir(output, overwrite=False)

    assert prepare_output_dir(output, overwrite=True) == output
    assert (output / "existing.txt").is_file()


def test_prepare_output_dir_rejects_file_path(tmp_path: Path) -> None:
    output = tmp_path / "not_a_directory"
    output.write_text("file", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not a directory"):
        prepare_output_dir(output, overwrite=True)


def test_evaluate_rejects_output_inside_dataset(tmp_path: Path) -> None:
    dataset = make_audio_dataset(tmp_path / "dataset")

    with pytest.raises(ValueError, match="outside the dataset"):
        evaluate(
            dataset_dir=dataset,
            split="test",
            method="wiener",
            output_dir=dataset / "experiments",
            limit=1,
            overwrite=False,
            n_fft=256,
            hop_length=64,
            noise_duration_sec=0.25,
            include_pesq=False,
            include_stoi=False,
        )


def test_main_reports_missing_dataset(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--dataset-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1


@pytest.mark.parametrize("method", ["wiener", "oracle-irm"])
def test_cli_smoke_writes_metrics(tmp_path: Path, method: str) -> None:
    pytest.importorskip("librosa")
    dataset = make_audio_dataset(tmp_path / "dataset")
    output = tmp_path / "results"

    exit_code = main(
        [
            "--dataset-dir",
            str(dataset),
            "--split",
            "test",
            "--method",
            method,
            "--output-dir",
            str(output),
            "--limit",
            "1",
        ]
    )

    assert exit_code == 0
    assert (output / "metrics.csv").is_file()
    assert (output / "config.json").is_file()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["count"] == 1
    assert summary["config"]["method"] == method
    assert summary["config"]["uses_clean_reference"] == (method == "oracle-irm")
    with (output / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["method"] == method
    assert float(rows[0]["input_snr_db"]) == pytest.approx(5.0, abs=0.5)
