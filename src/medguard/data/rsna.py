"""RSNA Pneumonia Detection localization dataset utilities.

This Phase 3 adapter is the active localization path. It treats RSNA pneumonia
boxes as engineering localization ground truth for lung opacity only; it does
not make clinical claims or score other NIH findings as localization targets.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from medguard.data.transforms import build_image_transform

PHASE = "3"

RSNA_LOCAL_LABELS = ["Lung Opacity"]

NIH_TO_RSNA_LABELS: dict[str, list[str]] = {
    "Atelectasis": [],
    "Cardiomegaly": [],
    "Effusion": [],
    "Infiltration": [],
    "Mass": [],
    "Nodule": [],
    "Pneumonia": ["Lung Opacity"],
    "Pneumothorax": [],
    "Consolidation": [],
    "Edema": [],
    "Emphysema": [],
    "Fibrosis": [],
    "Pleural_Thickening": [],
    "Hernia": [],
}


def is_available() -> bool:
    """Return whether the Phase 3 RSNA adapter is implemented."""

    return True


class RSNAPneumoniaDatasetUnavailableError(RuntimeError):
    """Raised when RSNA metadata or image files are unavailable."""


@dataclass(frozen=True)
class RSNABox:
    """One normalized RSNA pneumonia bounding-box annotation."""

    label: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class RSNARecord:
    """One RSNA image record with spatial metadata."""

    image_id: str
    patient_id: str
    split: str
    path: Path
    width: int
    height: int
    boxes: tuple[RSNABox, ...]
    detailed_class: str | None = None


class RSNAPneumoniaDataset(Dataset[dict[str, Any]]):
    """RSNA Pneumonia Detection localization dataset.

    The Kaggle challenge labels provide pneumonia/lung-opacity boxes in
    ``x, y, width, height`` pixel format. This loader normalizes them to
    ``[0, 1]`` ``xyxy`` coordinates and performs deterministic patient-level
    train/val/test assignment when no manifest is provided.
    """

    def __init__(
        self,
        root: str | Path,
        labels_csv: str | Path = "stage_2_train_labels.csv",
        split: str = "val",
        image_dir: str | Path = "stage_2_train_images",
        detailed_class_info_csv: str | Path | None = "stage_2_detailed_class_info.csv",
        manifest_csv: str | Path | None = None,
        labels: list[str] | None = None,
        include_negative: bool = False,
        val_fraction: float = 0.20,
        test_fraction: float = 0.00,
        split_seed: int = 2026,
        transform: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.labels_csv = _resolve_under_root(self.root, labels_csv)
        self.image_dir = Path(image_dir)
        self.detailed_class_info_csv = (
            _resolve_under_root(self.root, detailed_class_info_csv)
            if detailed_class_info_csv is not None
            else None
        )
        self.manifest_csv = (
            _resolve_under_root(self.root, manifest_csv) if manifest_csv is not None else None
        )
        self.labels = labels or list(RSNA_LOCAL_LABELS)
        self.split = split
        self.include_negative = include_negative
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.split_seed = split_seed
        self.transform = transform
        self.records = self._load_records()
        if not self.records:
            raise RSNAPneumoniaDatasetUnavailableError(
                f"No RSNA pneumonia records found for split {split!r} under {self.root}."
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        split: str | None = None,
    ) -> RSNAPneumoniaDataset:
        """Build a dataset from ``configs/grounding_rsna.yaml`` style config."""

        data_cfg = config.get("data", {})
        split_cfg = config.get("split", {})
        preprocessing_cfg = config.get("preprocessing", {})
        transform = None
        if preprocessing_cfg:
            transform = build_image_transform(config, train=False)
        return cls(
            root=data_cfg.get("root", "data/rsna"),
            labels_csv=data_cfg.get("labels_csv", "stage_2_train_labels.csv"),
            image_dir=data_cfg.get("image_dir", "stage_2_train_images"),
            detailed_class_info_csv=data_cfg.get(
                "detailed_class_info_csv",
                "stage_2_detailed_class_info.csv",
            ),
            manifest_csv=data_cfg.get("manifest_csv"),
            labels=list(data_cfg.get("labels", RSNA_LOCAL_LABELS)),
            include_negative=bool(data_cfg.get("include_negative", False)),
            val_fraction=float(split_cfg.get("val_fraction", 0.20)),
            test_fraction=float(split_cfg.get("test_fraction", 0.00)),
            split_seed=int(config.get("seed", 2026)),
            split=split or split_cfg.get("active", split_cfg.get("eval", "val")),
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = _read_image(record.path)
        transformed = self.transform(image) if self.transform is not None else _pil_to_tensor(image)
        boxes = torch.tensor([box.bbox for box in record.boxes], dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        labels = [box.label for box in record.boxes]
        label_indices = torch.tensor(
            [self.labels.index(label) for label in labels],
            dtype=torch.long,
        )
        binary_label = torch.tensor(float(len(record.boxes) > 0), dtype=torch.float32)
        return {
            "image": transformed,
            "label": binary_label,
            "image_id": record.image_id,
            "patient_id": record.patient_id,
            "path": str(record.path),
            "boxes": boxes,
            "box_labels": labels,
            "box_label_indices": label_indices,
            "width": record.width,
            "height": record.height,
            "split": record.split,
            "detailed_class": record.detailed_class,
        }

    def _load_records(self) -> list[RSNARecord]:
        if not self.labels_csv.exists():
            raise RSNAPneumoniaDatasetUnavailableError(
                f"RSNA labels CSV not found: {self.labels_csv}"
            )
        rows = _read_csv(self.labels_csv)
        class_info = _read_class_info(self.detailed_class_info_csv)
        manifest_splits = _read_manifest_splits(self.manifest_csv)
        grouped: dict[str, list[RSNABox]] = {}
        target_by_patient: dict[str, bool] = {}

        for row in rows:
            patient_id = _first_present(row, "patientId", "patient_id", "image_id")
            target = _target_is_positive(_first_present(row, "Target", "target", default="0"))
            target_by_patient[patient_id] = target_by_patient.get(patient_id, False) or target
            if not target:
                grouped.setdefault(patient_id, [])
                continue
            label = "Lung Opacity"
            if label not in self.labels:
                continue
            path = _resolve_image_path(self.root, self.image_dir, patient_id, row)
            width, height = _image_size(path, row)
            box = _rsna_box_from_row(row, width=width, height=height)
            grouped.setdefault(patient_id, []).append(RSNABox(label=label, bbox=box))

        records: list[RSNARecord] = []
        for patient_id in sorted(grouped):
            boxes = tuple(grouped[patient_id])
            if not boxes and not self.include_negative:
                continue
            split = manifest_splits.get(
                patient_id,
                _deterministic_split(
                    patient_id=patient_id,
                    seed=self.split_seed,
                    val_fraction=self.val_fraction,
                    test_fraction=self.test_fraction,
                ),
            )
            if self.split != "all" and split != self.split:
                continue
            path = _resolve_image_path(self.root, self.image_dir, patient_id, {})
            if not path.exists():
                raise RSNAPneumoniaDatasetUnavailableError(f"RSNA image file not found: {path}")
            width, height = _image_size(path)
            detailed_class = class_info.get(patient_id)
            if (
                target_by_patient.get(patient_id, False)
                and detailed_class not in {None, "Lung Opacity"}
            ):
                raise RSNAPneumoniaDatasetUnavailableError(
                    "RSNA positive target without Lung Opacity class info for "
                    f"{patient_id!r}: {detailed_class!r}"
                )
            records.append(
                RSNARecord(
                    image_id=patient_id,
                    patient_id=patient_id,
                    split=split,
                    path=path,
                    width=width,
                    height=height,
                    boxes=boxes,
                    detailed_class=detailed_class,
                )
            )
        return records


def dataset_available(config: Mapping[str, Any]) -> bool:
    """Return whether configured RSNA labels and at least one image exist."""

    data_cfg = config.get("data", {})
    root = Path(data_cfg.get("root", "data/rsna"))
    labels_csv = _resolve_under_root(root, data_cfg.get("labels_csv", "stage_2_train_labels.csv"))
    if not labels_csv.exists():
        return False
    try:
        rows = _read_csv(labels_csv)
    except (OSError, KeyError, RSNAPneumoniaDatasetUnavailableError):
        return False
    image_dir = Path(data_cfg.get("image_dir", "stage_2_train_images"))
    for row in rows[:25]:
        try:
            patient_id = _first_present(row, "patientId", "patient_id", "image_id")
        except KeyError:
            continue
        if _resolve_image_path(root, image_dir, patient_id, row).exists():
            return True
    return False


def normalize_rsna_bbox(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Normalize one RSNA ``x, y, width, height`` pixel box to ``xyxy``."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")
    if width <= 0 or height <= 0:
        raise ValueError("RSNA box width and height must be positive.")
    return _clip_and_validate_box(
        (
            x / image_width,
            y / image_height,
            (x + width) / image_width,
            (y + height) / image_height,
        )
    )


def rsna_labels_for_nih_label(nih_label: str) -> list[str]:
    """Return RSNA localization labels that correspond to one NIH label."""

    return list(NIH_TO_RSNA_LABELS.get(nih_label, []))


def read_rsna_image(path: str | Path) -> Image.Image:
    """Read one RSNA image as grayscale PIL image."""

    return _read_image(Path(path))


def _rsna_box_from_row(
    row: Mapping[str, str],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    return normalize_rsna_bbox(
        x=float(_first_present(row, "x", "x_min", "xmin")),
        y=float(_first_present(row, "y", "y_min", "ymin")),
        width=float(_first_present(row, "width", "w", "bbox_width")),
        height=float(_first_present(row, "height", "h", "bbox_height")),
        image_width=width,
        image_height=height,
    )


def _target_is_positive(value: str) -> bool:
    return str(value).strip() in {"1", "1.0", "true", "True", "TRUE"}


def _deterministic_split(
    patient_id: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> str:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.")
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode()).hexdigest()
    bucket = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + val_fraction:
        return "val"
    return "train"


def _read_class_info(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    rows = _read_csv(path)
    info: dict[str, str] = {}
    for row in rows:
        patient_id = _first_present(row, "patientId", "patient_id", "image_id")
        info[patient_id] = _first_present(row, "class", "class_name", "label")
    return info


def _read_manifest_splits(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    rows = _read_csv(path)
    splits: dict[str, str] = {}
    for row in rows:
        patient_id = _first_present(row, "patientId", "patient_id", "image_id")
        splits[patient_id] = _first_present(row, "split", "Split", "set")
    return splits


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_under_root(root: Path, path: str | Path | None) -> Path:
    if path is None:
        raise RSNAPneumoniaDatasetUnavailableError("Required RSNA path is missing from config.")
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


def _resolve_image_path(
    root: Path,
    image_dir: Path,
    patient_id: str,
    row: Mapping[str, str],
) -> Path:
    configured = _optional_first_present(row, "path", "image_path", "filepath")
    if configured:
        candidate = Path(configured)
        return candidate if candidate.is_absolute() else root / candidate
    candidates = [
        root / image_dir / patient_id,
        root / image_dir / f"{patient_id}.dcm",
        root / image_dir / f"{patient_id}.dicom",
        root / image_dir / f"{patient_id}.png",
        root / image_dir / f"{patient_id}.jpg",
        root / f"{patient_id}.dcm",
        root / f"{patient_id}.dicom",
        root / f"{patient_id}.png",
        root / f"{patient_id}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _image_size(path: Path, row: Mapping[str, str] | None = None) -> tuple[int, int]:
    width = _optional_first_present(row or {}, "image_width", "Columns")
    height = _optional_first_present(row or {}, "image_height", "Rows")
    if width is not None and height is not None:
        return int(float(width)), int(float(height))
    if not path.exists():
        raise RSNAPneumoniaDatasetUnavailableError(
            f"Cannot infer RSNA image dimensions; file missing: {path}"
        )
    if path.suffix.lower() in {".dcm", ".dicom"}:
        try:
            import pydicom

            metadata = pydicom.dcmread(path, stop_before_pixels=True)
            return int(metadata.Columns), int(metadata.Rows)
        except Exception as exc:  # pragma: no cover - depends on local DICOM files.
            raise RSNAPneumoniaDatasetUnavailableError(
                f"Failed to read DICOM metadata: {path}"
            ) from exc
    with Image.open(path) as image:
        return image.size


def _read_image(path: Path) -> Image.Image:
    if path.suffix.lower() in {".dcm", ".dicom"}:
        try:
            import pydicom

            dataset = pydicom.dcmread(path)
            array = dataset.pixel_array.astype(np.float32)
            array -= float(array.min())
            max_value = float(array.max())
            if max_value > 0:
                array /= max_value
            return Image.fromarray((array * 255.0).astype(np.uint8), mode="L")
        except Exception as exc:  # pragma: no cover - depends on local DICOM files.
            raise RSNAPneumoniaDatasetUnavailableError(
                f"Failed to read DICOM image: {path}"
            ) from exc
    with Image.open(path) as image:
        return image.convert("L")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _clip_and_validate_box(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = (float(np.clip(value, 0.0, 1.0)) for value in bbox)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid normalized RSNA bbox after clipping: {bbox}")
    return x_min, y_min, x_max, y_max


def _first_present(row: Mapping[str, str], *keys: str, default: str | None = None) -> str:
    value = _optional_first_present(row, *keys)
    if value is None:
        if default is not None:
            return default
        raise KeyError(f"Missing required column. Tried: {', '.join(keys)}")
    return value


def _optional_first_present(row: Mapping[str, str], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    return None
