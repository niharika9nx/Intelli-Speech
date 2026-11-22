"""GMM noise-type classifier.

One diagonal-covariance Gaussian mixture is trained per background-noise class on
log-mel + delta frames of noisy mixtures. An utterance is scored by the mean frame
log-likelihood under each class model; a softmax with a calibrated temperature turns
those scores into a class posterior. The posterior conditions the ZF-Net mask
estimator and is reported to the user as the detected background noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

GMM_FORMAT_VERSION = 1


class GMMDependencyError(RuntimeError):
    """Raised when scikit-learn or joblib is not installed."""


def _import_sklearn_mixture():
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as exc:
        raise GMMDependencyError(
            "scikit-learn is required for the GMM noise classifier "
            "(pip install scikit-learn)."
        ) from exc
    return GaussianMixture


def _import_joblib():
    try:
        import joblib
    except Exception as exc:
        raise GMMDependencyError("joblib is required to save/load the GMM.") from exc
    return joblib


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    return weights / weights.sum()


@dataclass
class NoiseGMM:
    """Per-class GMMs over frame features, with a temperature-scaled posterior."""

    n_components: int = 16
    covariance_type: str = "diag"
    max_frames_per_class: int = 60000
    random_state: int = 0
    temperature: float = 1.0
    classes: list[str] = field(default_factory=list)
    models: dict[str, object] = field(default_factory=dict)
    feature_dim: int | None = None

    @property
    def is_fitted(self) -> bool:
        return bool(self.classes) and all(name in self.models for name in self.classes)

    def fit(self, features_by_class: Mapping[str, Sequence[np.ndarray]]) -> "NoiseGMM":
        """Fit one GMM per class from lists of (frames, dim) feature arrays."""
        GaussianMixture = _import_sklearn_mixture()
        if len(features_by_class) < 2:
            raise ValueError("At least two noise classes are required.")

        rng = np.random.default_rng(self.random_state)
        classes = sorted(features_by_class)
        models: dict[str, object] = {}
        feature_dim: int | None = None
        for name in classes:
            value = features_by_class[name]
            if isinstance(value, np.ndarray) and value.ndim == 2:
                value = [value]
            arrays = [np.asarray(item, dtype=np.float64) for item in value]
            arrays = [item for item in arrays if item.ndim == 2 and item.shape[0] > 0]
            if not arrays:
                raise ValueError(f"No feature frames for class {name!r}.")
            frames = np.concatenate(arrays, axis=0)
            if feature_dim is None:
                feature_dim = frames.shape[1]
            elif frames.shape[1] != feature_dim:
                raise ValueError("All classes must share the same feature dimension.")
            if not np.isfinite(frames).all():
                raise ValueError(f"Non-finite features for class {name!r}.")
            if frames.shape[0] > self.max_frames_per_class:
                index = rng.choice(frames.shape[0], self.max_frames_per_class, replace=False)
                frames = frames[index]
            components = max(1, min(int(self.n_components), frames.shape[0]))
            model = GaussianMixture(
                n_components=components,
                covariance_type=self.covariance_type,
                reg_covar=1e-4,
                max_iter=200,
                random_state=self.random_state,
            )
            model.fit(frames)
            models[name] = model

        self.classes = classes
        self.models = models
        self.feature_dim = feature_dim
        return self

    def class_scores(self, frames: np.ndarray) -> np.ndarray:
        """Mean frame log-likelihood under each class model, in ``classes`` order."""
        if not self.is_fitted:
            raise RuntimeError("NoiseGMM is not fitted.")
        array = np.asarray(frames, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("Expected a non-empty (frames, dim) feature array.")
        if self.feature_dim is not None and array.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, received {array.shape[1]}."
            )
        return np.array(
            [float(np.mean(self.models[name].score_samples(array))) for name in self.classes]
        )

    def predict_proba(self, frames: np.ndarray) -> np.ndarray:
        """Class posterior for one utterance."""
        scores = self.class_scores(frames)
        return _softmax(scores / max(float(self.temperature), 1e-6)).astype(np.float32)

    def predict(self, frames: np.ndarray) -> str:
        return self.classes[int(np.argmax(self.class_scores(frames)))]

    def calibrate_temperature(
        self,
        utterances: Sequence[np.ndarray],
        labels: Sequence[str],
        *,
        grid: Sequence[float] | None = None,
    ) -> float:
        """Choose the softmax temperature minimizing validation negative log-likelihood."""
        if len(utterances) != len(labels) or not utterances:
            raise ValueError("utterances and labels must be non-empty and aligned.")
        candidates = list(grid) if grid is not None else list(np.geomspace(0.05, 20.0, 40))
        scores = np.stack([self.class_scores(item) for item in utterances])
        targets = np.array([self.classes.index(label) for label in labels])
        best_temperature, best_nll = float(self.temperature), np.inf
        for temperature in candidates:
            scaled = scores / float(temperature)
            scaled -= scaled.max(axis=1, keepdims=True)
            log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
            nll = float(-log_probs[np.arange(len(targets)), targets].mean())
            if nll < best_nll:
                best_temperature, best_nll = float(temperature), nll
        self.temperature = best_temperature
        return best_temperature

    def save(self, path: str | Path) -> Path:
        joblib = _import_joblib()
        if not self.is_fitted:
            raise RuntimeError("Refusing to save an unfitted NoiseGMM.")
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format_version": GMM_FORMAT_VERSION,
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
                "max_frames_per_class": self.max_frames_per_class,
                "random_state": self.random_state,
                "temperature": self.temperature,
                "classes": list(self.classes),
                "models": self.models,
                "feature_dim": self.feature_dim,
            },
            output,
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "NoiseGMM":
        joblib = _import_joblib()
        _import_sklearn_mixture()
        payload = joblib.load(Path(path).expanduser())
        if payload.get("format_version") != GMM_FORMAT_VERSION:
            raise ValueError(f"Unsupported GMM checkpoint format in {path}.")
        return cls(
            n_components=int(payload["n_components"]),
            covariance_type=str(payload["covariance_type"]),
            max_frames_per_class=int(payload["max_frames_per_class"]),
            random_state=int(payload["random_state"]),
            temperature=float(payload["temperature"]),
            classes=list(payload["classes"]),
            models=dict(payload["models"]),
            feature_dim=payload["feature_dim"],
        )
