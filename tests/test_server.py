from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from server.app import GMM_PATH_ENV, ZFNET_PATH_ENV, create_app

NOISE_PROBS = {"STRAFFIC": 0.7, "TBUS": 0.2, "OOFFICE": 0.1}


@dataclass(frozen=True)
class FakeOutput:
    waveform: np.ndarray
    sample_rate: int
    noise_type: str
    noise_probs: dict[str, float]


class FakeEnhancer:
    model_version = "fake-v0"
    sample_rate = 16000

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def enhance(self, waveform: np.ndarray) -> FakeOutput:
        self.calls.append(waveform)
        return FakeOutput(
            waveform=(0.5 * waveform).astype(np.float32),
            sample_rate=16000,
            noise_type="STRAFFIC",
            noise_probs=dict(NOISE_PROBS),
        )


def make_wav(sample_rate: int = 16000, seconds: float = 0.5, channels: int = 1) -> bytes:
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    tone = (0.25 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def post_wav(client: TestClient, data: bytes):
    return client.post("/enhance", files={"file": ("input.wav", data, "audio/wav")})


def test_health_reports_loaded_model() -> None:
    client = TestClient(create_app(enhancer=FakeEnhancer()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "fake-v0",
    }


def test_enhance_returns_decodable_wav() -> None:
    enhancer = FakeEnhancer()
    client = TestClient(create_app(enhancer=enhancer))

    response = post_wav(client, make_wav(seconds=0.5))

    assert response.status_code == 200
    body = response.json()
    assert body["sample_rate"] == 16000
    assert body["noise_type"] == "STRAFFIC"
    assert body["noise_probs"] == pytest.approx(NOISE_PROBS)
    assert body["model_version"] == "fake-v0"
    assert body["duration_sec"] == pytest.approx(0.5)
    assert body["processing_ms"] >= 0
    audio, rate = sf.read(io.BytesIO(base64.b64decode(body["audio_wav_base64"])))
    assert rate == 16000
    assert audio.ndim == 1
    assert audio.shape[0] == 8000
    assert np.max(np.abs(audio)) == pytest.approx(0.125, abs=2e-3)


def test_enhance_resamples_stereo_44k_to_16k_mono() -> None:
    enhancer = FakeEnhancer()
    client = TestClient(create_app(enhancer=enhancer))

    response = post_wav(client, make_wav(sample_rate=44100, seconds=1.0, channels=2))

    assert response.status_code == 200
    (received,) = enhancer.calls
    assert received.ndim == 1
    assert received.dtype == np.float32
    assert received.shape[0] == 16000
    audio, rate = sf.read(io.BytesIO(base64.b64decode(response.json()["audio_wav_base64"])))
    assert rate == 16000
    assert audio.shape == (16000,)


def test_enhance_rejects_empty_upload() -> None:
    enhancer = FakeEnhancer()
    client = TestClient(create_app(enhancer=enhancer))

    response = post_wav(client, b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
    assert enhancer.calls == []


def test_enhance_rejects_audio_longer_than_limit() -> None:
    client = TestClient(create_app(enhancer=FakeEnhancer()))

    response = post_wav(client, make_wav(sample_rate=8000, seconds=61.0))

    assert response.status_code == 400


def test_enhance_without_model_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GMM_PATH_ENV, raising=False)
    monkeypatch.delenv(ZFNET_PATH_ENV, raising=False)
    client = TestClient(create_app())

    health = client.get("/health").json()
    response = post_wav(client, make_wav())

    assert health["model_loaded"] is False
    assert health["model_version"] is None
    assert response.status_code == 503
    assert GMM_PATH_ENV in response.json()["detail"]
