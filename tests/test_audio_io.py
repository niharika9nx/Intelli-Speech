from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from intelli_speech.audio_io import (
    AudioLoadError,
    load_audio,
    resolve_metadata_audio_path,
    validate_mono_16khz,
    write_audio,
)


def test_resolve_relative_audio_path(tmp_path: Path) -> None:
    (tmp_path / "clean").mkdir()
    audio_path = tmp_path / "clean" / "sample.wav"
    audio_path.touch()

    resolved = resolve_metadata_audio_path(tmp_path, "clean/sample.wav")

    assert resolved == audio_path


def test_resolve_absolute_audio_path(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.touch()

    resolved = resolve_metadata_audio_path(tmp_path, audio_path)

    assert resolved == audio_path


def test_resolve_missing_audio_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_metadata_audio_path(tmp_path, "missing.wav")


def test_validate_mono_16khz_accepts_mono(tmp_path: Path) -> None:
    waveform = np.array([0.0, 0.25, -0.25], dtype=np.float32)

    validated = validate_mono_16khz(waveform, 16000)

    assert validated.dtype == np.float32
    assert np.array_equal(validated, waveform)


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "message"),
    [
        (np.array([], dtype=np.float32), 16000, "empty"),
        (np.ones((4, 2), dtype=np.float32), 16000, "mono"),
        (np.array([0.0, np.nan], dtype=np.float32), 16000, "non-finite"),
        (np.ones(4, dtype=np.float32), 8000, "16000"),
    ],
)
def test_validate_rejects_invalid_audio(
    waveform: np.ndarray,
    sample_rate: int,
    message: str,
) -> None:
    with pytest.raises(AudioLoadError, match=message):
        validate_mono_16khz(waveform, sample_rate)


def test_load_and_write_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "output" / "enhanced.wav"
    waveform = np.array([0.0, 0.1, -0.1, 0.0], dtype=np.float32)
    sf.write(str(source), waveform, 16000, subtype="PCM_16")

    loaded, sample_rate = load_audio(source)
    returned_path = write_audio(output, loaded, sample_rate)

    reread, reread_rate = sf.read(str(returned_path), dtype="float32")
    assert reread_rate == 16000
    assert len(reread) == len(waveform)
    assert np.isfinite(reread).all()
