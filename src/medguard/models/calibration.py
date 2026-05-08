"""Per-class calibration methods for multi-label CXR logits."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

PHASE = "2"
EPSILON = 1e-7


def is_available() -> bool:
    """Return whether calibration logic is implemented."""
    return True


class Calibrator(Protocol):
    """Per-class probability calibrator over multi-label sigmoid outputs."""

    classes: list[str]
    method: str

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        image_ids: Sequence[str] | None = None,
    ) -> None:
        """Fit from validation-set raw logits and binary labels."""

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities in ``[0, 1]`` with shape ``[N, C]``."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable parameter dump."""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Calibrator:
        """Rebuild the calibrator from a JSON-serializable payload."""


@dataclass
class TemperatureScalingCalibrator:
    """Independent temperature scaling for every class."""

    classes: list[str]
    method: str = "temperature"
    temperatures: np.ndarray = field(init=False)
    degenerate_classes: list[str] = field(default_factory=list)
    n_fit_samples: int | None = None
    fit_split_hash: str | None = None
    _fit_called: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.temperatures = np.ones(len(self.classes), dtype=np.float64)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        image_ids: Sequence[str] | None = None,
    ) -> None:
        """Fit one positive temperature per class on validation logits only."""
        logits, labels = _validate_arrays(logits, labels, len(self.classes))
        self._record_fit_metadata(logits, image_ids)
        temperatures = np.ones(len(self.classes), dtype=np.float64)
        degenerate: list[str] = []

        for class_index, class_name in enumerate(self.classes):
            target = labels[:, class_index]
            if _is_degenerate(target):
                degenerate.append(class_name)
                continue
            log_temperature = torch.zeros(1, dtype=torch.float64, requires_grad=True)
            z = torch.as_tensor(logits[:, class_index], dtype=torch.float64)
            y = torch.as_tensor(target, dtype=torch.float64)
            optimizer = torch.optim.LBFGS(
                [log_temperature],
                lr=0.25,
                max_iter=50,
                line_search_fn="strong_wolfe",
            )

            def closure(
                optimizer: torch.optim.LBFGS = optimizer,
                log_temperature: torch.Tensor = log_temperature,
                z: torch.Tensor = z,
                y: torch.Tensor = y,
            ) -> torch.Tensor:
                optimizer.zero_grad()
                temperature = torch.exp(log_temperature).clamp_min(EPSILON)
                loss = F.binary_cross_entropy_with_logits(z / temperature, y, reduction="sum")
                loss.backward()
                return loss

            optimizer.step(closure)
            temperatures[class_index] = float(torch.exp(log_temperature.detach()).item())

        self.temperatures = np.clip(temperatures, EPSILON, None)
        self.degenerate_classes = degenerate

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply ``sigmoid(logit / T_c)`` per class."""
        logits = _validate_logits(logits, len(self.classes))
        probs = _sigmoid(logits / self.temperatures.reshape(1, -1))
        return np.clip(probs, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable parameter dump."""
        return {
            "method": self.method,
            "classes": self.classes,
            "temperatures": [float(value) for value in self.temperatures.tolist()],
            "degenerate_classes": list(self.degenerate_classes),
            "n_fit_samples": self.n_fit_samples,
            "fit_split_hash": self.fit_split_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TemperatureScalingCalibrator:
        """Rebuild a temperature calibrator from ``to_dict`` output."""
        calibrator = cls(classes=list(payload["classes"]))
        calibrator.temperatures = np.asarray(payload["temperatures"], dtype=np.float64)
        calibrator.degenerate_classes = list(payload.get("degenerate_classes", []))
        calibrator.n_fit_samples = payload.get("n_fit_samples")
        calibrator.fit_split_hash = payload.get("fit_split_hash")
        calibrator._fit_called = True
        return calibrator

    def _record_fit_metadata(
        self,
        logits: np.ndarray,
        image_ids: Sequence[str] | None,
    ) -> None:
        _assert_not_fit(self._fit_called)
        self._fit_called = True
        self.n_fit_samples = int(logits.shape[0])
        self.fit_split_hash = hash_image_ids(image_ids) if image_ids is not None else None


@dataclass
class IsotonicCalibrator:
    """Per-class isotonic regression over uncalibrated sigmoid probabilities."""

    classes: list[str]
    method: str = "isotonic"
    thresholds: list[dict[str, list[float]] | None] = field(default_factory=list)
    degenerate_classes: list[str] = field(default_factory=list)
    n_fit_samples: int | None = None
    fit_split_hash: str | None = None
    _fit_called: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.thresholds:
            self.thresholds = [None for _ in self.classes]

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        image_ids: Sequence[str] | None = None,
    ) -> None:
        """Fit one isotonic model per non-degenerate class."""
        logits, labels = _validate_arrays(logits, labels, len(self.classes))
        self._record_fit_metadata(logits, image_ids)
        probs = _sigmoid(logits)
        thresholds: list[dict[str, list[float]] | None] = []
        degenerate: list[str] = []

        for class_index, class_name in enumerate(self.classes):
            target = labels[:, class_index]
            if _is_degenerate(target):
                thresholds.append(None)
                degenerate.append(class_name)
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(probs[:, class_index], target)
            thresholds.append(
                {
                    "x": [float(value) for value in iso.X_thresholds_.tolist()],
                    "y": [float(value) for value in iso.y_thresholds_.tolist()],
                }
            )

        self.thresholds = thresholds
        self.degenerate_classes = degenerate

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply fitted isotonic maps, with identity passthrough for degenerate classes."""
        logits = _validate_logits(logits, len(self.classes))
        probs = _sigmoid(logits)
        calibrated = np.empty_like(probs, dtype=np.float64)
        for class_index, threshold in enumerate(self.thresholds):
            if threshold is None:
                calibrated[:, class_index] = probs[:, class_index]
                continue
            x = np.asarray(threshold["x"], dtype=np.float64)
            y = np.asarray(threshold["y"], dtype=np.float64)
            calibrated[:, class_index] = np.interp(
                probs[:, class_index],
                x,
                y,
                left=float(y[0]),
                right=float(y[-1]),
            )
        return np.clip(calibrated, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable parameter dump."""
        return {
            "method": self.method,
            "classes": self.classes,
            "thresholds": self.thresholds,
            "isotonic_n_thresholds": [
                0 if threshold is None else len(threshold["x"]) for threshold in self.thresholds
            ],
            "degenerate_classes": list(self.degenerate_classes),
            "n_fit_samples": self.n_fit_samples,
            "fit_split_hash": self.fit_split_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IsotonicCalibrator:
        """Rebuild an isotonic calibrator from ``to_dict`` output."""
        calibrator = cls(classes=list(payload["classes"]))
        calibrator.thresholds = list(payload.get("thresholds", []))
        calibrator.degenerate_classes = list(payload.get("degenerate_classes", []))
        calibrator.n_fit_samples = payload.get("n_fit_samples")
        calibrator.fit_split_hash = payload.get("fit_split_hash")
        calibrator._fit_called = True
        return calibrator

    def _record_fit_metadata(
        self,
        logits: np.ndarray,
        image_ids: Sequence[str] | None,
    ) -> None:
        _assert_not_fit(self._fit_called)
        self._fit_called = True
        self.n_fit_samples = int(logits.shape[0])
        self.fit_split_hash = hash_image_ids(image_ids) if image_ids is not None else None


@dataclass
class PlattCalibrator:
    """Per-class logistic regression calibration over raw logits."""

    classes: list[str]
    method: str = "platt"
    a: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    degenerate_classes: list[str] = field(default_factory=list)
    n_fit_samples: int | None = None
    fit_split_hash: str | None = None
    _fit_called: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.a = np.ones(len(self.classes), dtype=np.float64)
        self.b = np.zeros(len(self.classes), dtype=np.float64)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        image_ids: Sequence[str] | None = None,
    ) -> None:
        """Fit one nearly unregularized logistic regression per class."""
        logits, labels = _validate_arrays(logits, labels, len(self.classes))
        self._record_fit_metadata(logits, image_ids)
        a = np.ones(len(self.classes), dtype=np.float64)
        b = np.zeros(len(self.classes), dtype=np.float64)
        degenerate: list[str] = []

        for class_index, class_name in enumerate(self.classes):
            target = labels[:, class_index]
            if _is_degenerate(target):
                degenerate.append(class_name)
                continue
            model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
            model.fit(logits[:, class_index].reshape(-1, 1), target.astype(int))
            a[class_index] = float(model.coef_[0, 0])
            b[class_index] = float(model.intercept_[0])

        self.a = a
        self.b = b
        self.degenerate_classes = degenerate

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply ``sigmoid(A_c * z_c + B_c)`` per class."""
        logits = _validate_logits(logits, len(self.classes))
        return np.clip(_sigmoid(logits * self.a.reshape(1, -1) + self.b.reshape(1, -1)), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable parameter dump."""
        return {
            "method": self.method,
            "classes": self.classes,
            "A": [float(value) for value in self.a.tolist()],
            "B": [float(value) for value in self.b.tolist()],
            "degenerate_classes": list(self.degenerate_classes),
            "n_fit_samples": self.n_fit_samples,
            "fit_split_hash": self.fit_split_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlattCalibrator:
        """Rebuild a Platt calibrator from ``to_dict`` output."""
        calibrator = cls(classes=list(payload["classes"]))
        calibrator.a = np.asarray(payload["A"], dtype=np.float64)
        calibrator.b = np.asarray(payload["B"], dtype=np.float64)
        calibrator.degenerate_classes = list(payload.get("degenerate_classes", []))
        calibrator.n_fit_samples = payload.get("n_fit_samples")
        calibrator.fit_split_hash = payload.get("fit_split_hash")
        calibrator._fit_called = True
        return calibrator

    def _record_fit_metadata(
        self,
        logits: np.ndarray,
        image_ids: Sequence[str] | None,
    ) -> None:
        _assert_not_fit(self._fit_called)
        self._fit_called = True
        self.n_fit_samples = int(logits.shape[0])
        self.fit_split_hash = hash_image_ids(image_ids) if image_ids is not None else None


def build_calibrator(method: str, classes: list[str]) -> Calibrator:
    """Build a calibrator by method name."""
    method = method.lower()
    if method == "temperature":
        return TemperatureScalingCalibrator(classes=classes)
    if method == "isotonic":
        return IsotonicCalibrator(classes=classes)
    if method == "platt":
        return PlattCalibrator(classes=classes)
    raise ValueError(f"Unsupported calibration method: {method}")


def save_calibrator(calibrator: Calibrator, path: str | Path) -> None:
    """Persist a calibrator pickle and adjacent JSON audit payload."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(calibrator, handle)
    output_path.with_suffix(".json").write_text(json.dumps(calibrator.to_dict(), indent=2))


def load_calibrator(path: str | Path) -> Calibrator:
    """Load a persisted calibrator pickle."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def hash_image_ids(image_ids: Sequence[str]) -> str:
    """Return a deterministic SHA-256 hash for an image-id split."""
    payload = "\n".join(sorted(str(image_id) for image_id in image_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_not_fit(fit_called: bool) -> None:
    if fit_called:
        raise RuntimeError("Calibrator.fit() may be called exactly once per instance.")


def _validate_arrays(
    logits: np.ndarray,
    labels: np.ndarray,
    expected_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    validated_logits = _validate_logits(logits, expected_classes)
    validated_labels = np.asarray(labels, dtype=np.float64)
    if validated_labels.shape != validated_logits.shape:
        raise ValueError(
            f"Labels shape {validated_labels.shape} does not match logits shape "
            f"{validated_logits.shape}."
        )
    return validated_logits, validated_labels


def _validate_logits(logits: np.ndarray, expected_classes: int) -> np.ndarray:
    validated = np.asarray(logits, dtype=np.float64)
    if validated.ndim != 2 or validated.shape[1] != expected_classes:
        raise ValueError(
            f"Expected logits with shape [N, {expected_classes}], got {validated.shape}."
        )
    return validated


def _is_degenerate(target: np.ndarray) -> bool:
    unique = np.unique(target)
    return unique.size < 2


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return np.clip(result, EPSILON, 1.0 - EPSILON)
