"""Local OOD gates for Phase 4 safety-aware CXR inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

PHASE = "4"

DEFAULT_CONFIG = {
    "blank_std_threshold": 0.02,
    "noise_low_freq_ratio_min": 0.40,
    "natural_color_max_mean": 5.0,
    "natural_edge_chi2_max": 1.0,
    "view_classifier_path": None,
}


@dataclass(frozen=True)
class OODDecision:
    """Single-image OOD gate decision."""

    accepted: bool
    reason: str
    score: dict[str, float]
    warning_only: bool = False


def is_available() -> bool:
    """Return whether Phase 4 OOD logic is implemented."""

    return True


def detect_ood(
    image: str | Path | np.ndarray | Image.Image,
    config: Mapping[str, Any] | None = None,
    view_classifier: Callable[[np.ndarray], str] | None = None,
) -> OODDecision:
    """Run local OOD checks in the fixed Phase 4 order."""

    cfg = {**DEFAULT_CONFIG, **dict(config or {})}
    gray, color_diff = _load_image_arrays(image)
    scores: dict[str, float] = {}

    std = float(np.std(gray))
    scores["std"] = std
    if std < float(cfg["blank_std_threshold"]):
        return OODDecision(False, "ood_blank_image", scores)

    low_freq_ratio = _low_frequency_ratio(gray)
    scores["low_freq_ratio"] = low_freq_ratio
    if low_freq_ratio < float(cfg["noise_low_freq_ratio_min"]):
        return OODDecision(False, "ood_corrupted_image", scores)

    mean_color_diff = float(color_diff)
    edge_chi2 = _edge_prior_chi2(gray)
    scores["mean_color_channel_diff_8bit"] = mean_color_diff
    scores["edge_prior_chi2"] = edge_chi2
    if (
        mean_color_diff > float(cfg["natural_color_max_mean"])
        and edge_chi2 > float(cfg["natural_edge_chi2_max"])
    ):
        return OODDecision(False, "ood_natural_image", scores)

    view_classifier_path = cfg.get("view_classifier_path")
    if view_classifier_path and view_classifier is None:
        raise RuntimeError(
            "view_classifier_path is set in config but no callable was wired into "
            "detect_ood; either provide one or null out the path."
        )
    if view_classifier is not None:
        view = view_classifier(gray)
        scores["view_classifier_available"] = 1.0
        if str(view).lower() in {"lateral", "unknown"}:
            return OODDecision(True, "ood_unsupported_view", scores, warning_only=True)
    else:
        scores["view_classifier_available"] = 0.0

    return OODDecision(True, "", scores)


def load_ood_config(path: str | Path = "configs/ood.yaml") -> dict[str, Any]:
    """Load OOD thresholds from YAML, falling back to safe defaults."""

    config_path = Path(path)
    if not config_path.exists():
        return dict(DEFAULT_CONFIG)
    payload = yaml.safe_load(config_path.read_text()) or {}
    return {**DEFAULT_CONFIG, **payload.get("ood", {})}


def _load_image_arrays(image: str | Path | np.ndarray | Image.Image) -> tuple[np.ndarray, float]:
    if isinstance(image, str | Path):
        pil = Image.open(image)
    elif isinstance(image, Image.Image):
        pil = image
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            gray = _normalize_to_unit(array)
            return gray, 0.0
        if array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
            rgb = _normalize_to_unit(array[..., :3])
            return _rgb_to_gray(rgb), _mean_color_diff_8bit(rgb)
        raise ValueError("image must be a path, PIL image, 2D array, or RGB array.")

    if pil.mode in {"RGB", "RGBA"}:
        rgb = _normalize_to_unit(np.asarray(pil.convert("RGB")))
        return _rgb_to_gray(rgb), _mean_color_diff_8bit(rgb)
    gray = _normalize_to_unit(np.asarray(pil.convert("L")))
    return gray, 0.0


def _normalize_to_unit(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("image must not be empty.")
    if np.nanmax(arr) > 1.0:
        arr = arr / 255.0
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )


def _mean_color_diff_8bit(rgb: np.ndarray) -> float:
    diff = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
    return float(np.mean(diff) * 255.0)


def _low_frequency_ratio(gray: np.ndarray) -> float:
    centered = gray - float(np.mean(gray))
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    energy = np.abs(spectrum) ** 2
    total = float(np.sum(energy))
    if total <= 1e-12:
        return 1.0
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt(((yy - cy) / max(height, 1)) ** 2 + ((xx - cx) / max(width, 1)) ** 2)
    low = radius <= 0.25
    return float(np.sum(energy[low]) / total)


def _edge_prior_chi2(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray.astype(np.float32))
    magnitude = np.sqrt(gx**2 + gy**2)
    if float(np.sum(magnitude)) <= 1e-12:
        hist = np.zeros(16, dtype=np.float64)
    else:
        height, width = gray.shape
        yy, xx = np.indices(gray.shape)
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        dist = np.sqrt(((yy - cy) / max(height, 1)) ** 2 + ((xx - cx) / max(width, 1)) ** 2)
        hist, _ = np.histogram(dist, bins=16, range=(0.0, 0.75), weights=magnitude)
        hist = hist.astype(np.float64)
        hist = hist / max(float(np.sum(hist)), 1e-12)
    prior = _load_edge_prior()
    return float(np.sum((hist - prior) ** 2 / (prior + 1e-6)))


def _load_edge_prior() -> np.ndarray:
    path = Path(__file__).with_name("_cxr_edge_prior.npy")
    if path.exists():
        prior = np.load(path)
    else:
        prior = np.array(
            [0.02, 0.04, 0.08, 0.12, 0.16, 0.16, 0.13, 0.10,
             0.08, 0.05, 0.03, 0.015, 0.01, 0.005, 0.003, 0.002],
            dtype=np.float64,
        )
    prior = prior.astype(np.float64)
    return prior / max(float(np.sum(prior)), 1e-12)
