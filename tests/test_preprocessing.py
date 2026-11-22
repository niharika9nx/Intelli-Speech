from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from intelli_speech.preprocessing import (
    get_noise_segment,
    load_wav_resampled,
    measured_snr_db,
    mix_at_snr,
    scale_noise_to_snr,
    to_mono_16k,
    trim_silence,
    write_pcm16,
)


def test_to_mono_16k_resamples_stereo_samples_first() -> None:
    sample_rate = 8000
    time = np.arange(800, dtype=np.float32) / sample_rate
    left = np.sin(2 * np.pi * 220 * time)
    right = np.sin(2 * np.pi * 330 * time)
    stereo = np.stack([left, right], axis=1)

    mono = to_mono_16k(stereo, sample_rate)

    assert mono.shape == (1600,)
    assert mono.dtype == np.float32
    assert np.isfinite(mono).all()


def test_to_mono_16k_accepts_channels_first() -> None:
    sample_rate = 8000
    channels_first = np.ones((2, 800), dtype=np.float32) * 0.1

    mono = to_mono_16k(channels_first, sample_rate)

    assert mono.shape == (1600,)
    assert np.allclose(mono[100:-100], 0.1, atol=1e-3)


def test_to_mono_16k_rejects_invalid_audio() -> None:
    with pytest.raises(ValueError, match="empty"):
        to_mono_16k(np.array([], dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="non-finite"):
        to_mono_16k(np.array([0.0, np.nan], dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="sample_rate"):
        to_mono_16k(np.ones(10, dtype=np.float32), 0)


def test_get_noise_segment_is_deterministic_and_dc_free() -> None:
    noise = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    first = get_noise_segment(noise, 100, rng=np.random.default_rng(42))
    second = get_noise_segment(noise, 100, rng=np.random.default_rng(42))

    assert np.array_equal(first, second)
    assert first.shape == (100,)
    assert abs(float(np.mean(first))) < 1e-6


def test_get_noise_segment_tiles_short_noise() -> None:
    noise = np.linspace(0.1, 0.2, 10, dtype=np.float32)

    segment = get_noise_segment(noise, 25, rng=np.random.default_rng(1))

    assert segment.shape == (25,)


def test_get_noise_segment_rejects_silence() -> None:
    with pytest.raises(ValueError, match="silent"):
        get_noise_segment(np.zeros(100, dtype=np.float32), 50)


def test_scale_noise_to_snr_hits_target() -> None:
    rng = np.random.default_rng(7)
    clean = np.full(2000, 0.1, dtype=np.float32)
    noise = rng.normal(0, 0.01, clean.shape).astype(np.float32)

    mixture = scale_noise_to_snr(clean, noise, 5.0)

    assert mixture.shape == clean.shape
    assert measured_snr_db(clean, mixture) == pytest.approx(5.0, abs=0.05)


def test_mix_at_snr_applies_common_gain() -> None:
    rng = np.random.default_rng(11)
    clean = (0.5 * rng.normal(size=2000)).astype(np.float32)
    noise = rng.normal(size=2000).astype(np.float32)

    result = mix_at_snr(clean, noise, 10.0)

    assert result.clean.shape == result.mixture.shape == (2000,)
    assert result.measured_snr_db == pytest.approx(10.0, abs=0.05)
    assert 0 < result.common_gain <= 1.0
    assert float(np.max(np.abs(result.mixture))) <= 0.99 + 1e-6


def test_mix_at_snr_rejects_bad_peak_limit() -> None:
    clean = np.ones(10, dtype=np.float32) * 0.1
    noise = np.ones(10, dtype=np.float32) * 0.01

    with pytest.raises(ValueError, match="peak_limit"):
        mix_at_snr(clean, noise, 5.0, peak_limit=0)
    with pytest.raises(ValueError, match="peak_limit"):
        mix_at_snr(clean, noise, 5.0, peak_limit=1.1)


def test_write_pcm16_round_trip(tmp_path: Path) -> None:
    sample_rate = 16000
    time = np.arange(1600) / sample_rate
    waveform = (0.25 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    output = tmp_path / "nested" / "tone.wav"

    returned = write_pcm16(output, waveform, sample_rate=sample_rate)

    assert returned == output
    loaded, loaded_rate = sf.read(output, dtype="float32")
    assert loaded_rate == sample_rate
    assert len(loaded) == len(waveform)


def test_write_pcm16_rejects_clipping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="clipping"):
        write_pcm16(tmp_path / "clip.wav", np.array([0.0, 1.0], dtype=np.float32))


def test_load_wav_resampled(tmp_path: Path) -> None:
    source = tmp_path / "source_8k.wav"
    time = np.arange(800) / 8000
    sf.write(source, (0.2 * np.sin(2 * np.pi * 220 * time)), 8000)

    waveform = load_wav_resampled(source)

    assert waveform.shape == (1600,)
    assert waveform.dtype == np.float32


def test_trim_silence_preserves_speech() -> None:
    pytest.importorskip("librosa")
    sample_rate = 16000
    silence = np.zeros(sample_rate, dtype=np.float32)
    time = np.arange(sample_rate) / sample_rate
    speech = (0.3 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    waveform = np.concatenate([silence, speech, silence])

    trimmed = trim_silence(waveform, sample_rate=sample_rate)

    assert 0.5 * sample_rate <= len(trimmed) < len(waveform)
    assert np.isfinite(trimmed).all()


def test_trim_silence_rejects_empty() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        trim_silence(np.array([], dtype=np.float32))
