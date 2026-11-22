"""Local HTTP inference server for the Intelli-Speech enhancement model.

Run from the repository root:

    python -m uvicorn server.app:app --host 127.0.0.1 --port 8000

The model is loaded lazily on the first ``/enhance`` request from the
checkpoint paths in ``INTELLI_SPEECH_GMM_PATH`` and
``INTELLI_SPEECH_ZFNET_PATH``.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Allow `python -m uvicorn server.app:app` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import resample_poly

LOGGER = logging.getLogger("intelli_speech.server")

TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SEC = 60.0
GMM_PATH_ENV = "INTELLI_SPEECH_GMM_PATH"
ZFNET_PATH_ENV = "INTELLI_SPEECH_ZFNET_PATH"
CORS_ORIGINS_ENV = "INTELLI_SPEECH_CORS_ORIGINS"
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


class ModelUnavailableError(RuntimeError):
    """Raised when the enhancement model cannot be configured or loaded."""


def _cors_origins() -> list[str]:
    raw = os.environ.get(CORS_ORIGINS_ENV, DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _load_enhancer_from_env() -> Any:
    gmm_path = os.environ.get(GMM_PATH_ENV, "").strip()
    zfnet_path = os.environ.get(ZFNET_PATH_ENV, "").strip()
    if not gmm_path or not zfnet_path:
        raise ModelUnavailableError(
            "Enhancement model is not configured. Set "
            f"{GMM_PATH_ENV} and {ZFNET_PATH_ENV} to the checkpoint files."
        )
    missing = [
        name
        for name, value in ((GMM_PATH_ENV, gmm_path), (ZFNET_PATH_ENV, zfnet_path))
        if not Path(value).is_file()
    ]
    if missing:
        raise ModelUnavailableError(
            "Enhancement model checkpoint not found for: " + ", ".join(missing)
        )
    # Imported lazily so the server (and its tests) work without the model
    # dependencies when an enhancer is injected directly.
    from intelli_speech.enhancer import Enhancer

    return Enhancer.from_checkpoints(gmm_path, zfnet_path)


def _decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises several error types
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a readable WAV audio file."
        ) from exc
    if audio.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio contains no samples.")
    duration = audio.shape[0] / float(sample_rate)
    if duration > MAX_DURATION_SEC:
        raise HTTPException(
            status_code=400,
            detail=f"Audio is {duration:.1f} s long; the limit is {MAX_DURATION_SEC:.0f} s.",
        )
    mono = audio.mean(axis=1, dtype=np.float32)
    if not np.all(np.isfinite(mono)):
        raise HTTPException(status_code=400, detail="Uploaded audio contains invalid samples.")
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        up = TARGET_SAMPLE_RATE // divisor
        down = int(sample_rate) // divisor
        mono = resample_poly(mono, up, down).astype(np.float32)
    if mono.size == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio contains no samples.")
    return mono, TARGET_SAMPLE_RATE


def _encode_wav(waveform: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    buffer = io.BytesIO()
    sf.write(buffer, clipped, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def create_app(enhancer: Any | None = None) -> FastAPI:
    """Build the FastAPI application.

    If ``enhancer`` is None the model is loaded from environment variables on
    the first ``/enhance`` request.
    """

    application = FastAPI(title="Intelli-Speech inference server")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    state: dict[str, Any] = {"enhancer": enhancer}
    lock = threading.Lock()

    def get_enhancer() -> Any:
        if state["enhancer"] is not None:
            return state["enhancer"]
        with lock:
            if state["enhancer"] is None:
                try:
                    state["enhancer"] = _load_enhancer_from_env()
                except ModelUnavailableError as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                except Exception as exc:
                    LOGGER.exception("Failed to load enhancement model")
                    raise HTTPException(
                        status_code=503,
                        detail="Enhancement model failed to load. Check the server logs.",
                    ) from exc
                LOGGER.info(
                    "Loaded enhancement model %s",
                    getattr(state["enhancer"], "model_version", "unknown"),
                )
        return state["enhancer"]

    @application.get("/health")
    def health() -> dict[str, Any]:
        loaded = state["enhancer"]
        return {
            "status": "ok",
            "model_loaded": loaded is not None,
            "model_version": getattr(loaded, "model_version", None) if loaded else None,
        }

    @application.post("/enhance")
    def enhance(file: UploadFile = File(...)) -> dict[str, Any]:
        data = file.file.read()
        waveform, sample_rate = _decode_wav(data)
        model = get_enhancer()

        started = time.perf_counter()
        try:
            output = model.enhance(waveform)
        except Exception as exc:
            LOGGER.exception("Enhancement failed")
            raise HTTPException(status_code=500, detail="Enhancement failed.") from exc
        processing_ms = (time.perf_counter() - started) * 1000.0

        output_rate = int(getattr(output, "sample_rate", sample_rate))
        wav_bytes = _encode_wav(output.waveform, output_rate)
        duration_sec = waveform.shape[0] / float(sample_rate)
        LOGGER.info(
            "enhance duration=%.2fs noise_type=%s processing_ms=%.1f",
            duration_sec,
            output.noise_type,
            processing_ms,
        )
        return {
            "audio_wav_base64": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": output_rate,
            "noise_type": str(output.noise_type),
            "noise_probs": {str(k): float(v) for k, v in output.noise_probs.items()},
            "model_version": str(getattr(model, "model_version", "unknown")),
            "duration_sec": round(duration_sec, 3),
            "processing_ms": round(processing_ms, 1),
        }

    return application


app = create_app()
