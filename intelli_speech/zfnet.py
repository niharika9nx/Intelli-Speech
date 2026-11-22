"""ZF-Net-style convolutional mask estimator conditioned on the GMM noise posterior.

Architecture (v1)
-----------------
The encoder follows ZF-Net's five-convolution layout (7x7, 5x5, 3x3, 3x3, 3x3; ZF-Net
replaced AlexNet's 11x11/stride-4 first layer with a 7x7 filter). Downsampling is
applied along frequency only (stride (2, 1)) so every STFT frame keeps its own mask
column. The GMM noise posterior conditions the bottleneck through FiLM
(feature-wise affine modulation). A decoder with skip connections upsamples back to
the full frequency resolution and a sigmoid produces a ratio mask in [0, 1].

Input:  per-utterance-normalized log|X|, shape (batch, 1, freq_bins, frames)
Cond:   noise-class posterior, shape (batch, n_classes)
Output: mask, shape (batch, 1, freq_bins, frames)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ZFNET_FORMAT_VERSION = 1


class TorchDependencyError(RuntimeError):
    """Raised when PyTorch is not installed."""


def import_torch():
    try:
        import torch
    except Exception as exc:
        raise TorchDependencyError(
            "PyTorch is required for the ZF-Net model (preinstalled in Colab; "
            "locally: pip install torch --index-url https://download.pytorch.org/whl/cpu)."
        ) from exc
    return torch


@dataclass(frozen=True)
class ZFNetConfig:
    n_classes: int = 5
    freq_bins: int = 257
    base_channels: int = 16
    dropout: float = 0.1

    @property
    def channels(self) -> tuple[int, int, int, int, int]:
        c = int(self.base_channels)
        return (c, 2 * c, 4 * c, 4 * c, 4 * c)


torch = None
nn = None
try:  # Defer hard failure until the model is actually used.
    import torch  # type: ignore[no-redef]
    import torch.nn as nn  # type: ignore[no-redef]
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


if nn is not None:

    class ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: tuple[int, int]):
            super().__init__()
            self.conv = nn.Conv2d(
                in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2
            )
            self.norm = nn.BatchNorm2d(out_ch)

        def forward(self, x):
            return F.relu(self.norm(self.conv(x)))

    class FiLM(nn.Module):
        """Feature-wise linear modulation from the noise posterior."""

        def __init__(self, n_classes: int, channels: int):
            super().__init__()
            self.proj = nn.Linear(n_classes, 2 * channels)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

        def forward(self, x, cond):
            gamma, beta = self.proj(cond).chunk(2, dim=1)
            return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

    class ZFNetMask(nn.Module):
        def __init__(self, config: ZFNetConfig = ZFNetConfig()):
            super().__init__()
            self.config = config
            c1, c2, c3, c4, c5 = config.channels
            self.enc1 = ConvBlock(1, c1, 7, (2, 1))
            self.enc2 = ConvBlock(c1, c2, 5, (2, 1))
            self.enc3 = ConvBlock(c2, c3, 3, (2, 1))
            self.enc4 = ConvBlock(c3, c4, 3, (1, 1))
            self.enc5 = ConvBlock(c4, c5, 3, (1, 1))
            self.film = FiLM(config.n_classes, c5)
            self.dropout = nn.Dropout2d(config.dropout)
            self.dec3 = ConvBlock(c5 + c2, c3, 3, (1, 1))
            self.dec2 = ConvBlock(c3 + c1, c2, 3, (1, 1))
            self.dec1 = ConvBlock(c2 + 1, c1, 3, (1, 1))
            self.head = nn.Conv2d(c1, 1, kernel_size=1)

        @staticmethod
        def _up(x, like):
            return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)

        def forward(self, features, cond):
            e1 = self.enc1(features)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            bottleneck = self.enc5(self.enc4(e3))
            bottleneck = self.dropout(self.film(bottleneck, cond))
            d3 = self.dec3(torch.cat([self._up(bottleneck, e2), e2], dim=1))
            d2 = self.dec2(torch.cat([self._up(d3, e1), e1], dim=1))
            d1 = self.dec1(torch.cat([self._up(d2, features), features], dim=1))
            return torch.sigmoid(self.head(d1))

else:  # pragma: no cover

    class ZFNetMask:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            import_torch()


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------
# Torch STFT matching intelli_speech.stft (librosa) settings: periodic Hann window,
# center=True with zero ("constant") padding, no normalization.
# ---------------------------------------------------------------------------


def torch_stft(waveform, n_fft: int = 512, hop_length: int = 128):
    torch_module = import_torch()
    window = torch_module.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    return torch_module.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )


def torch_istft(spectrum, length: int, n_fft: int = 512, hop_length: int = 128):
    torch_module = import_torch()
    window = torch_module.hann_window(n_fft, device=spectrum.device)
    return torch_module.istft(
        spectrum,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        length=length,
    )


def torch_normalized_log_magnitude(spectrum, epsilon: float = 1e-6):
    """Batched equivalent of ``normalize_utterance(log_magnitude(X))``."""
    log_mag = torch.log(spectrum.abs() + epsilon)
    dims = tuple(range(1, log_mag.ndim))
    mean = log_mag.mean(dim=dims, keepdim=True)
    std = log_mag.std(dim=dims, keepdim=True, unbiased=False)
    return (log_mag - mean) / (std + 1e-5)


def mask_loss(predicted_mask, target_mask, noisy_magnitude, clean_magnitude, *, compress: float = 0.3):
    """Mask MSE plus compressed-magnitude MSE of the masked noisy spectrum."""
    mask_term = F.mse_loss(predicted_mask, target_mask)
    estimate = (predicted_mask * noisy_magnitude).clamp_min(1e-8) ** compress
    reference = clean_magnitude.clamp_min(1e-8) ** compress
    magnitude_term = F.mse_loss(estimate, reference)
    return mask_term + magnitude_term, {
        "mask_mse": float(mask_term.detach()),
        "magnitude_mse": float(magnitude_term.detach()),
    }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@dataclass
class ZFNetCheckpoint:
    model: Any
    config: ZFNetConfig
    noise_classes: list[str]
    n_fft: int
    hop_length: int
    sample_rate: int
    model_version: str
    extra: dict[str, Any] = field(default_factory=dict)


def save_zfnet_checkpoint(
    path: str | Path,
    model,
    *,
    noise_classes: list[str],
    n_fft: int,
    hop_length: int,
    sample_rate: int,
    model_version: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    torch_module = import_torch()
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(
        {
            "format_version": ZFNET_FORMAT_VERSION,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": asdict(model.config),
            "noise_classes": list(noise_classes),
            "n_fft": int(n_fft),
            "hop_length": int(hop_length),
            "sample_rate": int(sample_rate),
            "model_version": str(model_version),
            "extra": extra or {},
        },
        output,
    )
    return output


def load_zfnet_checkpoint(path: str | Path, device: str = "cpu") -> ZFNetCheckpoint:
    torch_module = import_torch()
    payload = torch_module.load(Path(path).expanduser(), map_location=device, weights_only=False)
    if payload.get("format_version") != ZFNET_FORMAT_VERSION:
        raise ValueError(f"Unsupported ZF-Net checkpoint format in {path}.")
    config = ZFNetConfig(**payload["config"])
    model = ZFNetMask(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return ZFNetCheckpoint(
        model=model,
        config=config,
        noise_classes=list(payload["noise_classes"]),
        n_fft=int(payload["n_fft"]),
        hop_length=int(payload["hop_length"]),
        sample_rate=int(payload["sample_rate"]),
        model_version=str(payload["model_version"]),
        extra=dict(payload.get("extra", {})),
    )


def predict_mask(model, normalized_log_mag: np.ndarray, noise_probs: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Run the model on one utterance; returns a float32 (freq, frames) mask."""
    torch_module = import_torch()
    features = torch_module.as_tensor(normalized_log_mag, dtype=torch_module.float32, device=device)
    cond = torch_module.as_tensor(noise_probs, dtype=torch_module.float32, device=device)
    with torch_module.no_grad():
        mask = model(features[None, None], cond[None])
    return mask[0, 0].cpu().numpy().astype(np.float32)
