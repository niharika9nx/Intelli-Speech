from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from intelli_speech.stft import (
    STFTConfig,
    STFTCheckpoint,
    compute_stft,
    inverse_stft,
    load_stft_checkpoint,
    reconstruction_error,
    save_stft_checkpoint,
    stft_round_trip,
)


def test_stft_config_defaults_and_validation() -> None:
    config = STFTConfig()

    assert config.freq_bins == 257
    assert config.sample_rate == 16000

    with pytest.raises(ValueError, match="n_fft"):
        STFTConfig(n_fft=0)
    with pytest.raises(ValueError, match="hop_length"):
        STFTConfig(hop_length=0)
    with pytest.raises(ValueError, match="hop_length"):
        STFTConfig(n_fft=256, hop_length=512)
    with pytest.raises(ValueError, match="sample_rate"):
        STFTConfig(sample_rate=0)
    with pytest.raises(ValueError, match="window"):
        STFTConfig(window="")


def test_reconstruction_error_known_values() -> None:
    original = np.ones(100, dtype=np.float32)
    reconstructed = original + 0.1

    report = reconstruction_error(original, reconstructed)

    assert report["rmse"] == pytest.approx(0.1)
    assert report["max_abs_error"] == pytest.approx(0.1)
    assert report["error_to_signal_snr_db"] == pytest.approx(20.0)


def test_reconstruction_error_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        reconstruction_error(np.ones(10), np.ones(11))


def test_save_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    clean = rng.normal(size=1000).astype(np.float32)
    noisy = (clean + 0.01 * rng.normal(size=1000)).astype(np.float32)
    clean_stft = rng.normal(size=(257, 10)) + 1j * rng.normal(size=(257, 10))
    noisy_stft = clean_stft + (0.01 * rng.normal(size=(257, 10)))
    checkpoint = STFTCheckpoint(
        clean=clean,
        noisy=noisy,
        clean_stft=clean_stft,
        noisy_stft=noisy_stft,
        sample_rate=16000,
        n_fft=512,
        hop_length=128,
    )
    path = tmp_path / "checkpoint.npz"

    returned = save_stft_checkpoint(path, checkpoint)
    loaded = load_stft_checkpoint(returned)

    assert returned == path
    assert np.array_equal(loaded.clean, clean)
    assert np.array_equal(loaded.noisy_stft, noisy_stft)
    assert loaded.config.n_fft == 512
    assert loaded.config.hop_length == 128


def test_load_checkpoint_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "missing.npz"
    np.savez(path, clean=np.ones(10), noisy=np.ones(10))

    with pytest.raises(ValueError, match="missing keys"):
        load_stft_checkpoint(path)


def test_load_checkpoint_bad_frequency_bins(tmp_path: Path) -> None:
    path = tmp_path / "bad_bins.npz"
    np.savez(
        path,
        clean=np.ones(10, dtype=np.float32),
        noisy=np.ones(10, dtype=np.float32),
        clean_stft=np.ones((10, 5), dtype=np.complex64),
        noisy_stft=np.ones((10, 5), dtype=np.complex64),
        sample_rate=np.asarray(16000),
        n_fft=np.asarray(512),
        hop_length=np.asarray(128),
    )

    with pytest.raises(ValueError, match="frequency bins"):
        load_stft_checkpoint(path)


def test_inverse_stft_rejects_bad_shape_without_librosa() -> None:
    with pytest.raises(ValueError, match="frequency bins"):
        inverse_stft(np.ones((100, 10), dtype=np.complex64), STFTConfig())


def test_compute_stft_round_trip() -> None:
    pytest.importorskip("librosa")
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    waveform = (0.3 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    config = STFTConfig(n_fft=256, hop_length=64)

    spectrum = compute_stft(waveform, config)
    reconstructed, report = stft_round_trip(waveform, config)

    assert spectrum.shape[0] == config.freq_bins
    assert reconstructed.shape == waveform.shape
    assert report["rmse"] < 1e-4
