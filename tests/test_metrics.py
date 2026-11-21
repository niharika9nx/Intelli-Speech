import numpy as np
import pytest

from intelli_speech.metrics import (
    clipping_report,
    compute_snr_db,
    peak_amplitude,
    rms_amplitude,
)


def test_compute_snr_db_known_values() -> None:
    signal = np.ones(100, dtype=np.float64)
    noise = np.ones(100, dtype=np.float64) * 0.1

    assert compute_snr_db(signal, noise) == pytest.approx(20.0)


def test_compute_snr_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        compute_snr_db(np.ones(2), np.ones(3))


def test_compute_snr_rejects_zero_noise() -> None:
    with pytest.raises(ValueError, match="Noise power"):
        compute_snr_db(np.ones(4), np.zeros(4))


def test_peak_and_rms_amplitudes() -> None:
    waveform = np.array([-0.5, 0.0, 0.25], dtype=np.float64)

    assert peak_amplitude(waveform) == 0.5
    assert rms_amplitude(waveform) == pytest.approx(
        np.sqrt((0.25 + 0.0625) / 3)
    )


def test_clipping_report() -> None:
    report = clipping_report(np.array([0.25, 1.0, -0.5]))

    assert report["peak_amplitude"] == 1.0
    assert report["full_scale_sample_count"] == 1
    assert report["clipping_detected"] is True
