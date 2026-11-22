"""End-to-end v1 enhancer: STFT -> GMM noise posterior + ZF-Net mask -> iSTFT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .baselines import apply_mask
from .features import log_magnitude, log_mel_frames, normalize_utterance
from .gmm_noise import NoiseGMM
from .stft import STFTConfig, compute_stft, inverse_stft
from .zfnet import load_zfnet_checkpoint, predict_mask


@dataclass(frozen=True)
class EnhancementOutput:
    waveform: np.ndarray
    sample_rate: int
    noise_type: str
    noise_probs: dict[str, float]


@dataclass(frozen=True)
class DetailedEnhancement:
    output: EnhancementOutput
    mask: np.ndarray
    enhanced_stft: np.ndarray


class Enhancer:
    """Load the trained GMM and ZF-Net checkpoints and enhance 16 kHz mono audio."""

    def __init__(self, gmm: NoiseGMM, zfnet_checkpoint, device: str = "cpu"):
        if list(gmm.classes) != list(zfnet_checkpoint.noise_classes):
            raise ValueError(
                "GMM classes and ZF-Net conditioning classes differ: "
                f"{gmm.classes} vs {zfnet_checkpoint.noise_classes}."
            )
        self.gmm = gmm
        self.checkpoint = zfnet_checkpoint
        self.model = zfnet_checkpoint.model
        self.device = device
        self.sample_rate = int(zfnet_checkpoint.sample_rate)
        self.model_version = zfnet_checkpoint.model_version
        self.stft_config = STFTConfig(
            n_fft=zfnet_checkpoint.n_fft,
            hop_length=zfnet_checkpoint.hop_length,
            sample_rate=self.sample_rate,
        )

    @classmethod
    def from_checkpoints(
        cls,
        gmm_path: str | Path,
        zfnet_path: str | Path,
        device: str = "cpu",
    ) -> "Enhancer":
        return cls(NoiseGMM.load(gmm_path), load_zfnet_checkpoint(zfnet_path, device), device)

    def enhance_detailed(self, waveform: np.ndarray) -> DetailedEnhancement:
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError("Expected a non-empty mono waveform.")
        if not np.isfinite(audio).all():
            raise ValueError("Waveform contains non-finite samples.")

        noisy_stft = compute_stft(audio, self.stft_config)
        probs = self.gmm.predict_proba(log_mel_frames(noisy_stft, self.stft_config))
        features = normalize_utterance(log_magnitude(noisy_stft))
        mask = np.clip(predict_mask(self.model, features, probs, self.device), 0.0, 1.0)
        enhanced_stft = apply_mask(noisy_stft, mask)
        enhanced = inverse_stft(enhanced_stft, self.stft_config, length=audio.size)

        noise_probs = {name: float(p) for name, p in zip(self.gmm.classes, probs)}
        noise_type = max(noise_probs, key=noise_probs.get)
        return DetailedEnhancement(
            output=EnhancementOutput(
                waveform=enhanced,
                sample_rate=self.sample_rate,
                noise_type=noise_type,
                noise_probs=noise_probs,
            ),
            mask=mask,
            enhanced_stft=enhanced_stft,
        )

    def enhance(self, waveform: np.ndarray) -> EnhancementOutput:
        return self.enhance_detailed(waveform).output
