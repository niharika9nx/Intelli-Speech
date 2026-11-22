from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from intelli_speech.frcrn import (
    ClearVoiceDependencyError,
    FRCRNConfig,
    enhance_file,
    inspect_model,
    normalize_waveform,
)
from scripts.run_frcrn import main


class FakeFRCRNModel:
    model_names = ("FRCRN_SE_16K",)

    def __call__(self, *, input_path: str, online_write: bool = False) -> np.ndarray:
        assert online_write is False
        audio, _ = sf.read(input_path, dtype="float32")
        return audio


def test_normalize_waveform_handles_common_return_shapes() -> None:
    waveform = np.linspace(-0.5, 0.5, 160, dtype=np.float32)

    assert np.array_equal(normalize_waveform(waveform), waveform)
    assert normalize_waveform(waveform.reshape(-1, 1)).shape == waveform.shape
    assert normalize_waveform(waveform.reshape(1, -1)).shape == waveform.shape
    assert np.array_equal(
        normalize_waveform({"enhanced": waveform}),
        waveform,
    )
    assert np.array_equal(
        normalize_waveform([waveform]),
        waveform,
    )


def test_normalize_waveform_downmixes_two_channels() -> None:
    channels_first = np.stack([np.ones(80), np.zeros(80)], axis=0)

    result = normalize_waveform(channels_first)

    np.testing.assert_allclose(result, 0.5, atol=1e-6)


def test_normalize_waveform_rejects_nonfinite_and_empty() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        normalize_waveform(np.array([0.0, np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="empty waveform"):
        normalize_waveform(np.array([], dtype=np.float32))


def test_frcrn_config_requires_16khz() -> None:
    with pytest.raises(ValueError, match="16 kHz"):
        FRCRNConfig(sample_rate=8000)


def test_inspect_model_and_enhance_file_with_fake_model(tmp_path: Path) -> None:
    sample_rate = 16000
    time = np.arange(1600) / sample_rate
    waveform = (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    audio_path = tmp_path / "noisy.wav"
    sf.write(audio_path, waveform, sample_rate, subtype="PCM_16")
    model = FakeFRCRNModel()

    info = inspect_model(model)
    result = enhance_file(model, audio_path, sample_rate=16000)

    assert info["model_names"] == ("FRCRN_SE_16K",)
    assert result.enhanced.shape == waveform.shape
    assert result.input_path == audio_path
    assert result.model_names == ("FRCRN_SE_16K",)
    assert result.output_peak <= 1.0
    assert np.isfinite(result.enhanced).all()


def test_enhance_file_rejects_missing_model_and_missing_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "missing.wav"
    with pytest.raises(ValueError, match="not initialized"):
        enhance_file(None, audio_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        enhance_file(FakeFRCRNModel(), audio_path)


def test_build_frcrn_model_reports_missing_dependency() -> None:
    if importlib.util.find_spec("clearvoice") is None:
        with pytest.raises(ClearVoiceDependencyError, match="not importable"):
            from intelli_speech.frcrn import build_frcrn_model

            build_frcrn_model()
    else:
        pytest.skip("clearvoice is installed; dependency guard is not exercised.")


def test_run_frcrn_main_reports_missing_dataset(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--dataset-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1
