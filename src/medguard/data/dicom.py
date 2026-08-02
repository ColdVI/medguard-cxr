"""Shared, provenance-preserving DICOM preprocessing for chest radiographs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.pixels import apply_modality_lut, apply_voi_lut

PHASE = "academic_v2"


@dataclass(frozen=True)
class DICOMPreprocessResult:
    """Normalized model input and non-identifying transformation provenance."""

    image: np.ndarray
    provenance: dict[str, Any]


def is_available() -> bool:
    return True


def preprocess_dicom(
    source: str | Path | Dataset,
    *,
    channels: int = 3,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> DICOMPreprocessResult:
    """Decode a DICOM into a finite float32 ``[C,H,W]`` tensor contract.

    Patient-identifying tags are intentionally absent from provenance.
    """

    if channels not in {1, 3}:
        raise ValueError("DICOM output channels must be 1 or 3")
    dataset = (
        source
        if isinstance(source, Dataset)
        else pydicom.dcmread(str(Path(source)), force=False)
    )
    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim != 2:
        raise ValueError(f"Expected a single-frame grayscale DICOM, got shape {pixels.shape}")
    pixels = np.asarray(apply_modality_lut(pixels, dataset), dtype=np.float64)
    voi_applied = False
    if hasattr(dataset, "VOILUTSequence") or (
        hasattr(dataset, "WindowCenter") and hasattr(dataset, "WindowWidth")
    ):
        try:
            pixels = np.asarray(apply_voi_lut(pixels, dataset), dtype=np.float64)
            voi_applied = True
        except (TypeError, ValueError, IndexError):
            voi_applied = False
    finite = np.isfinite(pixels)
    if not finite.any():
        raise ValueError("DICOM contains no finite pixel values")
    if not finite.all():
        fill = float(np.median(pixels[finite]))
        pixels = np.where(finite, pixels, fill)
    photometric = str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2"))
    inverted = photometric.upper() == "MONOCHROME1"
    if inverted:
        pixels = pixels.max() + pixels.min() - pixels
    low, high = np.percentile(pixels, [lower_percentile, upper_percentile])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("DICOM clipping bounds are non-finite")
    if high <= low:
        normalized = np.zeros_like(pixels, dtype=np.float32)
    else:
        normalized = np.clip(pixels, low, high)
        normalized = ((normalized - low) / (high - low)).astype(np.float32)
    image = normalized[None, ...]
    if channels == 3:
        image = np.repeat(image, 3, axis=0)
    if not np.isfinite(image).all():
        raise ValueError("Normalized DICOM model input contains NaN or Inf")
    provenance = {
        "pipeline": "medguard_dicom_academic_v2",
        "rows": int(normalized.shape[0]),
        "columns": int(normalized.shape[1]),
        "channels": channels,
        "rescale_slope": float(getattr(dataset, "RescaleSlope", 1.0)),
        "rescale_intercept": float(getattr(dataset, "RescaleIntercept", 0.0)),
        "voi_lut_applied": voi_applied,
        "photometric_interpretation": photometric,
        "monochrome1_inverted": inverted,
        "clip_percentiles": [lower_percentile, upper_percentile],
        "clip_values": [float(low), float(high)],
    }
    return DICOMPreprocessResult(image=image, provenance=provenance)
