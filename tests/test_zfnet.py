"""Tests for intelli_speech.zfnet."""
from __future__ import annotations

import numpy as np
import pytest

from intelli_speech.zfnet import ZFNetMask, ZFNetConfig, torch_stft, torch_istft, mask_loss


def test_zfnet_mask_forward_shape_and_range() -> None:
    """ZFNetMask forward on (2,1,257,37) features and (2,3) cond returns (2,1,257,37) in [0,1]."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    # features: (batch=2, channels=1, freq=257, frames=37)
    features = torch.randn(2, 1, 257, 37)
    # cond: (batch=2, n_classes=3)
    cond = torch.randn(2, 3)
    config = ZFNetConfig(n_classes=3, base_channels=4)
    model = ZFNetMask(config)
    model.eval()
    with torch.no_grad():
        mask = model(features, cond)
    assert mask.shape == (2, 1, 257, 37), f"Expected (2,1,257,37), got {mask.shape}"
    assert mask.min().item() >= 0.0 and mask.max().item() <= 1.0


def test_torch_stft_matches_intelli_speech() -> None:
    """torch_stft on a random waveform matches intelli_speech.stft.compute_stft within 1e-4."""
    pytest.importorskip("librosa")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    waveform = (0.3 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)

    # intelli_speech compute_stft
    from intelli_speech.stft import STFTConfig, compute_stft
    config = STFTConfig(n_fft=512, hop_length=128)
    expected = compute_stft(waveform, config)

    # torch_stft
    w = torch.from_numpy(waveform).float()
    actual = torch_stft(w, n_fft=512, hop_length=128)
    actual_np = actual.detach().cpu().numpy()

    np.testing.assert_allclose(actual_np, expected, atol=1e-4)


def test_torch_istft_reconstructs_within_tolerance() -> None:
    """torch_istft(torch_stft(x)) reconstructs x within 1e-4."""
    pytest.importorskip("librosa")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    waveform = (0.3 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)

    w = torch.from_numpy(waveform).float()
    spectrum = torch_stft(w, n_fft=512, hop_length=128)
    reconstructed = torch_istft(spectrum, length=len(waveform), n_fft=512, hop_length=128)
    reconstructed_np = reconstructed.detach().cpu().numpy()

    np.testing.assert_allclose(waveform, reconstructed_np, atol=1e-4)


def test_mask_loss_decreases_with_adam_steps() -> None:
    """A few Adam steps on one fixed batch with mask_loss reduce the loss."""
    pytest.importorskip("torch")
    torch = pytest.importorskip("torch")

    # Create fixed predicted mask and targets
    predicted_mask = torch.rand(2, 1, 257, 10, dtype=torch.float32, requires_grad=True)
    target_mask = torch.rand(2, 1, 257, 10, dtype=torch.float32)
    noisy_magnitude = torch.rand(2, 1, 257, 10, dtype=torch.float32)
    clean_magnitude = torch.rand(2, 1, 257, 10, dtype=torch.float32)

    optimizer = torch.optim.Adam([predicted_mask], lr=0.1)

    loss_before, _ = mask_loss(predicted_mask, target_mask, noisy_magnitude, clean_magnitude)
    loss_before_val = loss_before.item()

    # A few optimization steps
    for _ in range(5):
        optimizer.zero_grad()
        loss, _ = mask_loss(predicted_mask, target_mask, noisy_magnitude, clean_magnitude)
        loss.backward()
        optimizer.step()

    loss_after, _ = mask_loss(predicted_mask, target_mask, noisy_magnitude, clean_magnitude)
    loss_after_val = loss_after.item()

    assert loss_after_val < loss_before_val, f"Loss did not decrease: {loss_before_val} -> {loss_after_val}"


def test_zfnet_checkpoint_save_load_roundtrip() -> None:
    """save_zfnet_checkpoint/load_zfnet_checkpoint round-trip gives same outputs in eval mode."""
    pytest.importorskip("torch")
    from intelli_speech.zfnet import save_zfnet_checkpoint, load_zfnet_checkpoint
    import tempfile, os

    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    config = ZFNetConfig(n_classes=3, base_channels=4)
    model = ZFNetMask(config)
    model.eval()

    # Dummy STFT config values
    n_fft = 512
    hop_length = 128
    sample_rate = 16000

    # Generate some dummy features and cond for forward check
    features = torch.randn(2, 1, 257, 37)
    cond = torch.randn(2, 3)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_zfnet_checkpoint(
            tmp_path,
            model,
            noise_classes=["noise_a", "noise_b", "noise_c"],
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate,
            model_version="v1",
        )

        loaded = load_zfnet_checkpoint(tmp_path)
        loaded.model.eval()

        with torch.no_grad():
            original_mask = model(features, cond).detach().cpu().numpy()
            loaded_mask = loaded.model(features, cond).detach().cpu().numpy()

        np.testing.assert_allclose(original_mask, loaded_mask, atol=1e-6)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)