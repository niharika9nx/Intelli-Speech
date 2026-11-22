"""Tests for intelli_speech.enhancer (end-to-end STFT -> GMM + ZF-Net -> iSTFT)."""
from __future__ import annotations

import numpy as np
import pytest


def _fitted_gmm(classes=("noise_a", "noise_b")):
    from intelli_speech.features import log_mel_frames
    from intelli_speech.gmm_noise import NoiseGMM
    from intelli_speech.stft import compute_stft

    rng = np.random.default_rng(0)
    features = {}
    for index, name in enumerate(classes):
        noise = rng.normal(size=16000).astype(np.float32) * 0.1
        if index:  # low-pass the second class so the spectra differ
            noise = np.convolve(noise, np.ones(8) / 8, mode="same").astype(np.float32)
        features[name] = log_mel_frames(compute_stft(noise))
    return NoiseGMM(n_components=2, random_state=0).fit(features)


def _tiny_model(n_classes: int):
    from intelli_speech.zfnet import ZFNetConfig, ZFNetMask

    model = ZFNetMask(ZFNetConfig(n_classes=n_classes, base_channels=2))
    model.eval()
    return model


def test_enhancer_round_trip_from_checkpoints(tmp_path) -> None:
    pytest.importorskip("librosa")
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")
    from intelli_speech.enhancer import EnhancementOutput, Enhancer
    from intelli_speech.zfnet import save_zfnet_checkpoint

    gmm = _fitted_gmm()
    gmm_path = gmm.save(tmp_path / "gmm.joblib")
    zfnet_path = save_zfnet_checkpoint(
        tmp_path / "zfnet.pt",
        _tiny_model(len(gmm.classes)),
        noise_classes=gmm.classes,
        n_fft=512,
        hop_length=128,
        sample_rate=16000,
        model_version="test-v1",
    )

    enhancer = Enhancer.from_checkpoints(gmm_path, zfnet_path)
    assert enhancer.model_version == "test-v1"
    assert enhancer.sample_rate == 16000

    t = np.arange(16000) / 16000
    waveform = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    output = enhancer.enhance(waveform)

    assert isinstance(output, EnhancementOutput)
    assert output.waveform.shape == waveform.shape
    assert np.isfinite(output.waveform).all()
    assert set(output.noise_probs) == set(gmm.classes)
    assert abs(sum(output.noise_probs.values()) - 1.0) < 1e-5
    assert output.noise_type in gmm.classes

    detailed = enhancer.enhance_detailed(waveform)
    assert detailed.mask.min() >= 0.0 and detailed.mask.max() <= 1.0


def test_enhancer_rejects_mismatched_classes() -> None:
    pytest.importorskip("librosa")
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")
    from intelli_speech.enhancer import Enhancer
    from intelli_speech.zfnet import ZFNetCheckpoint, ZFNetConfig

    gmm = _fitted_gmm()
    checkpoint = ZFNetCheckpoint(
        model=_tiny_model(2),
        config=ZFNetConfig(n_classes=2, base_channels=2),
        noise_classes=["other_a", "other_b"],
        n_fft=512,
        hop_length=128,
        sample_rate=16000,
        model_version="test-v1",
    )
    with pytest.raises(ValueError, match="classes differ"):
        Enhancer(gmm, checkpoint)


def test_enhancer_rejects_empty_waveform(tmp_path) -> None:
    pytest.importorskip("librosa")
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")
    from intelli_speech.enhancer import Enhancer
    from intelli_speech.zfnet import ZFNetCheckpoint, ZFNetConfig

    gmm = _fitted_gmm()
    checkpoint = ZFNetCheckpoint(
        model=_tiny_model(2),
        config=ZFNetConfig(n_classes=2, base_channels=2),
        noise_classes=list(gmm.classes),
        n_fft=512,
        hop_length=128,
        sample_rate=16000,
        model_version="test-v1",
    )
    with pytest.raises(ValueError):
        Enhancer(gmm, checkpoint).enhance(np.zeros(0, dtype=np.float32))
