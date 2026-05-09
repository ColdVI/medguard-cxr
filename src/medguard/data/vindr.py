"""DEFERRED FUTURE USE: optional VinDr-CXR loader, not the active pipeline.

RSNA Pneumonia Detection Challenge 2018 is the current localization/evidence
dataset. This module is retained so a future VinDr access grant can reactivate
the path deliberately, with a fresh data audit and owner approval.
"""

from __future__ import annotations

import csv
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
DEFERRED_FUTURE_WORK = True

VINDR_LOCAL_LABELS = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Clavicle fracture",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Enlarged PA",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Lung cavity",
    "Lung cyst",
    "Mediastinal shift",
    "Nodule/Mass",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Rib fracture",
    "Other lesion",
]

NIH_TO_VINDR_LABELS: dict[str, list[str]] = {
    "Atelectasis": ["Atelectasis"],
    "Cardiomegaly": ["Cardiomegaly"],
    "Effusion": ["Pleural effusion"],
    "Infiltration": ["Infiltration"],
    "Mass": ["Nodule/Mass"],
    "Nodule": ["Nodule/Mass"],
    "Pneumonia": [],
    "Pneumothorax": ["Pneumothorax"],
    "Consolidation": ["Consolidation"],
    "Edema": ["Edema"],
    "Emphysema": ["Emphysema"],
    "Fibrosis": ["Pulmonary fibrosis"],
    "Pleural_Thickening": ["Pleural thickening"],
    "Hernia": [],
}


def is_available() -> bool:
    """Return whether the Phase 3 VinDr implementation is available."""
    return True


class VinDrDatasetUnavailableError(RuntimeError):
    """Raised when VinDr metadata or image files are unavailable."""


@dataclass(frozen=True)
class VinDrBox:
    """One normalized bounding-box annotation."""

    label: str
    bbox: tuple[float, float, float, float]
    annotator_id: str | None = None


@dataclass(frozen=True)
class VinDrRecord:
    """One VinDr image record with spatial metadata."""

    image_id: str
    patient_id: str | None
    split: str
    path: Path
    width: int
    height: int
    boxes: tuple[VinDrBox, ...]


class VinDrCXRDataset(Dataset[dict[str, Any]]):
    """VinDr-CXR localization dataset with normalized boxes.

    The loader treats VinDr bounding boxes as the only localization ground truth.
    NIH image-level labels are intentionally not consumed here.
    """

    def __init__(
        self,
        root: str | Path,
        annotations_csv: str | Path,
        split: str,
        labels: list[str] | None = None,
        manifest_csv: str | Path | None = None,
        image_dir: str | Path = "images",
        bbox_format: str = "pixel_xyxy",
        consensus_strategy: str = "iou_merge",
        consensus_iou_threshold: float = 0.4,
        consensus_min_annotators: int = 1,
        consensus_grid_size: int = 256,
        allow_empty_annotations: bool = False,
        transform: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.annotations_csv = _resolve_under_root(self.root, annotations_csv)
        self.manifest_csv = (
            _resolve_under_root(self.root, manifest_csv) if manifest_csv is not None else None
        )
        self.image_dir = Path(image_dir)
        self.split = split
        self.labels = labels or list(VINDR_LOCAL_LABELS)
        self.bbox_format = bbox_format
        self.consensus_strategy = consensus_strategy
        self.consensus_iou_threshold = consensus_iou_threshold
        self.consensus_min_annotators = consensus_min_annotators
        self.consensus_grid_size = consensus_grid_size
        self.allow_empty_annotations = allow_empty_annotations
        self.transform = transform
        self.consensus_failure_by_image: dict[str, int] = {}
        self.consensus_failure_count = 0
        self.records = self._load_records()
        if not self.records:
            raise VinDrDatasetUnavailableError(
                f"No VinDr records found for split {split!r} under {self.root}."
            )

    @classmethod
    def from_config(cls, config: Mapping[str, Any], split: str | None = None) -> VinDrCXRDataset:
        """Build a dataset from ``configs/grounding_vindr.yaml`` style config."""
        data_cfg = config.get("data", {})
        split_cfg = config.get("split", {})
        preprocessing_cfg = config.get("preprocessing", {})
        consensus_cfg = data_cfg.get("consensus", {})
        transform = None
        if preprocessing_cfg:
            transform = build_image_transform(config, train=False)
        return cls(
            root=data_cfg.get("root", "data/vindr"),
            annotations_csv=data_cfg.get("annotations_csv", "annotations.csv"),
            manifest_csv=data_cfg.get("manifest_csv"),
            image_dir=data_cfg.get("image_dir", "images"),
            labels=list(data_cfg.get("labels", VINDR_LOCAL_LABELS)),
            split=split or split_cfg.get("active", split_cfg.get("eval", "test")),
            bbox_format=str(data_cfg.get("bbox_format", "pixel_xyxy")),
            consensus_strategy=str(consensus_cfg.get("strategy", "iou_merge")),
            consensus_iou_threshold=float(consensus_cfg.get("iou_threshold", 0.4)),
            consensus_min_annotators=int(consensus_cfg.get("min_annotators", 1)),
            consensus_grid_size=int(consensus_cfg.get("grid_size", 256)),
            allow_empty_annotations=bool(data_cfg.get("allow_empty_annotations", False)),
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = _read_image(record.path)
        transformed = self.transform(image) if self.transform is not None else _pil_to_tensor(image)
        boxes = torch.tensor([box.bbox for box in record.boxes], dtype=torch.float32)
        labels = [box.label for box in record.boxes]
        label_indices = torch.tensor(
            [self.labels.index(label) for label in labels],
            dtype=torch.long,
        )
        return {
            "image": transformed,
            "image_id": record.image_id,
            "patient_id": record.patient_id,
            "path": str(record.path),
            "boxes": boxes,
            "box_labels": labels,
            "box_label_indices": label_indices,
            "width": record.width,
            "height": record.height,
            "split": record.split,
        }

    def _load_records(self) -> list[VinDrRecord]:
        if not self.annotations_csv.exists():
            raise VinDrDatasetUnavailableError(
                f"VinDr annotation CSV not found: {self.annotations_csv}"
            )
        annotations, consensus_failures = _read_annotations(
            root=self.root,
            annotations_csv=self.annotations_csv,
            image_dir=self.image_dir,
            labels=self.labels,
            bbox_format=self.bbox_format,
            consensus_strategy=self.consensus_strategy,
            consensus_iou_threshold=self.consensus_iou_threshold,
            consensus_min_annotators=self.consensus_min_annotators,
            consensus_grid_size=self.consensus_grid_size,
        )
        self.consensus_failure_by_image = consensus_failures
        self.consensus_failure_count = sum(consensus_failures.values())
        manifest_rows = _read_manifest(self.manifest_csv)
        if not manifest_rows:
            manifest_rows = _manifest_from_annotations(annotations)

        records: list[VinDrRecord] = []
        for row in manifest_rows:
            row_split = _first_present(row, "split", "Split", "set", default="")
            if row_split and row_split != self.split:
                continue
            image_id = _first_present(row, "image_id", "imageId", "image", "image_id_png")
            patient_id = _optional_first_present(row, "patient_id", "Patient ID", "patientId")
            path = _resolve_image_path(self.root, self.image_dir, image_id, row)
            if not path.exists():
                raise VinDrDatasetUnavailableError(f"VinDr image file not found: {path}")
            width, height = _image_size(path, row)
            boxes = annotations.get(image_id, [])
            if not boxes and not self.allow_empty_annotations:
                if image_id in consensus_failures:
                    raise VinDrDatasetUnavailableError(
                        "No VinDr consensus annotations survived for image "
                        f"{image_id!r}; failed class groups={consensus_failures[image_id]}."
                    )
                raise VinDrDatasetUnavailableError(
                    f"Missing VinDr annotations for image {image_id!r}; not silently skipping."
                )
            records.append(
                VinDrRecord(
                    image_id=image_id,
                    patient_id=patient_id,
                    split=row_split or self.split,
                    path=path,
                    width=width,
                    height=height,
                    boxes=tuple(boxes),
                )
            )
        return records


def dataset_available(config: Mapping[str, Any]) -> bool:
    """Return whether configured VinDr annotations and at least one image exist."""
    data_cfg = config.get("data", {})
    root = Path(data_cfg.get("root", "data/vindr"))
    annotations = _resolve_under_root(root, data_cfg.get("annotations_csv", "annotations.csv"))
    if not annotations.exists():
        return False
    try:
        rows = _read_csv(annotations)
    except (OSError, KeyError, VinDrDatasetUnavailableError):
        return False
    for row in rows[:10]:
        try:
            image_id = _first_present(row, "image_id", "imageId", "image", "image_id_png")
        except KeyError:
            continue
        path = _resolve_image_path(root, Path(data_cfg.get("image_dir", "images")), image_id, row)
        if path.exists():
            return True
    return False


def normalize_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    bbox_format: str = "pixel_xyxy",
) -> tuple[float, float, float, float]:
    """Normalize a bbox to ``[0, 1]`` xyxy coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    x_min, y_min, x_max, y_max = bbox
    if bbox_format == "normalized_xyxy":
        normalized = (x_min, y_min, x_max, y_max)
    elif bbox_format == "pixel_xyxy":
        normalized = (x_min / width, y_min / height, x_max / width, y_max / height)
    else:
        raise ValueError(f"Unsupported bbox format: {bbox_format}")
    return _clip_and_validate_box(normalized)


def vindr_labels_for_nih_label(nih_label: str) -> list[str]:
    """Return VinDr localization labels that correspond to one NIH label."""

    return list(NIH_TO_VINDR_LABELS.get(nih_label, []))


def consensus_box_from_annotations(
    boxes: list[VinDrBox],
    strategy: str = "iou_merge",
    iou_threshold: float = 0.4,
    min_annotators: int = 1,
    grid_size: int = 256,
) -> VinDrBox | None:
    """Build one consensus box from same-image, same-class radiologist boxes."""

    if min_annotators < 1:
        raise ValueError("min_annotators must be >= 1.")
    if len(boxes) < min_annotators:
        return None
    labels = {box.label for box in boxes}
    if len(labels) != 1:
        raise ValueError("Consensus boxes require one label at a time.")

    if strategy == "iou_merge":
        bbox = _iou_merge_consensus(
            boxes,
            iou_threshold=iou_threshold,
            min_annotators=min_annotators,
        )
    elif strategy == "majority_vote":
        bbox = _majority_vote_consensus(
            boxes,
            min_annotators=min_annotators,
            grid_size=grid_size,
        )
    elif strategy in {"none", "raw"}:
        bbox = boxes[0].bbox
    else:
        raise ValueError(f"Unsupported VinDr consensus strategy: {strategy}")

    if bbox is None:
        return None
    return VinDrBox(
        label=boxes[0].label,
        bbox=bbox,
        annotator_id=f"consensus:{strategy}:n={len(boxes)}",
    )


def _read_annotations(
    root: Path,
    annotations_csv: Path,
    image_dir: Path,
    labels: list[str],
    bbox_format: str,
    consensus_strategy: str,
    consensus_iou_threshold: float,
    consensus_min_annotators: int,
    consensus_grid_size: int,
) -> tuple[dict[str, list[VinDrBox]], dict[str, int]]:
    rows = _read_csv(annotations_csv)
    raw_grouped: dict[tuple[str, str], list[VinDrBox]] = {}
    for row in rows:
        image_id = _first_present(row, "image_id", "imageId", "image", "image_id_png")
        label = _first_present(row, "class_name", "label", "finding", "class")
        if label not in labels:
            continue
        path = _resolve_image_path(root, image_dir, image_id, row)
        width, height = _image_size(path, row)
        raw_box = (
            float(_first_present(row, "x_min", "xmin", "x1")),
            float(_first_present(row, "y_min", "ymin", "y1")),
            float(_first_present(row, "x_max", "xmax", "x2")),
            float(_first_present(row, "y_max", "ymax", "y2")),
        )
        raw_grouped.setdefault((image_id, label), []).append(
            VinDrBox(
                label=label,
                bbox=normalize_bbox(raw_box, width=width, height=height, bbox_format=bbox_format),
                annotator_id=_optional_first_present(row, "rad_id", "annotator_id", "reader_id"),
            )
        )

    grouped: dict[str, list[VinDrBox]] = {}
    consensus_failures: dict[str, int] = {}
    for (image_id, _label), boxes in raw_grouped.items():
        consensus = consensus_box_from_annotations(
            boxes,
            strategy=consensus_strategy,
            iou_threshold=consensus_iou_threshold,
            min_annotators=consensus_min_annotators,
            grid_size=consensus_grid_size,
        )
        if consensus is not None:
            grouped.setdefault(image_id, []).append(consensus)
        else:
            consensus_failures[image_id] = consensus_failures.get(image_id, 0) + 1
    return grouped, consensus_failures


def _iou_merge_consensus(
    boxes: list[VinDrBox],
    iou_threshold: float,
    min_annotators: int,
) -> tuple[float, float, float, float] | None:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1].")

    clusters: list[list[VinDrBox]] = []
    for box in boxes:
        placed = False
        for cluster in clusters:
            if any(_box_iou(box.bbox, member.bbox) >= iou_threshold for member in cluster):
                cluster.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])

    eligible = [cluster for cluster in clusters if len(cluster) >= min_annotators]
    if not eligible:
        return None
    best_cluster = max(eligible, key=lambda cluster: (len(cluster), _union_area(cluster)))
    return _union_box([box.bbox for box in best_cluster])


def _majority_vote_consensus(
    boxes: list[VinDrBox],
    min_annotators: int,
    grid_size: int,
) -> tuple[float, float, float, float] | None:
    if grid_size < 8:
        raise ValueError("grid_size must be >= 8 for majority-vote consensus.")
    if len(boxes) < min_annotators:
        return None

    vote = np.zeros((grid_size, grid_size), dtype=np.int16)
    for box in boxes:
        x_min, y_min, x_max, y_max = box.bbox
        x0 = int(np.floor(x_min * grid_size))
        y0 = int(np.floor(y_min * grid_size))
        x1 = int(np.ceil(x_max * grid_size))
        y1 = int(np.ceil(y_max * grid_size))
        vote[
            np.clip(y0, 0, grid_size - 1) : np.clip(y1, 1, grid_size),
            np.clip(x0, 0, grid_size - 1) : np.clip(x1, 1, grid_size),
        ] += 1

    threshold = max(min_annotators, int(np.ceil(len(boxes) / 2)))
    ys, xs = np.where(vote >= threshold)
    if xs.size == 0:
        return None
    return _clip_and_validate_box(
        (
            float(xs.min() / grid_size),
            float(ys.min() / grid_size),
            float((xs.max() + 1) / grid_size),
            float((ys.max() + 1) / grid_size),
        )
    )


def _box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(box_a) + _box_area(box_b) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _box_area(box: tuple[float, float, float, float]) -> float:
    x_min, y_min, x_max, y_max = box
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


def _union_area(boxes: list[VinDrBox]) -> float:
    return _box_area(_union_box([box.bbox for box in boxes]))


def _union_box(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    arr = np.asarray(boxes, dtype=np.float64)
    return _clip_and_validate_box(
        (
            float(np.min(arr[:, 0])),
            float(np.min(arr[:, 1])),
            float(np.max(arr[:, 2])),
            float(np.max(arr[:, 3])),
        )
    )


def _read_manifest(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    return _read_csv(path)


def _manifest_from_annotations(annotations: Mapping[str, list[VinDrBox]]) -> list[dict[str, str]]:
    return [{"image_id": image_id, "split": "test"} for image_id in sorted(annotations)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_image_path(root: Path, image_dir: Path, image_id: str, row: Mapping[str, str]) -> Path:
    configured = _optional_first_present(row, "path", "image_path", "filepath")
    if configured:
        candidate = Path(configured)
        return candidate if candidate.is_absolute() else root / candidate
    candidates = [
        root / image_dir / image_id,
        root / image_dir / f"{image_id}.dicom",
        root / image_dir / f"{image_id}.dcm",
        root / image_dir / f"{image_id}.png",
        root / image_dir / f"{image_id}.jpg",
    ]
    split = _optional_first_present(row, "split", "Split", "set")
    if split:
        candidates.extend(
            [
                root / split / image_id,
                root / split / f"{image_id}.dicom",
                root / split / f"{image_id}.dcm",
                root / split / f"{image_id}.png",
                root / split / f"{image_id}.jpg",
            ]
        )
    candidates.extend(
        [
            root / f"{image_id}.dicom",
            root / f"{image_id}.dcm",
            root / f"{image_id}.png",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _image_size(path: Path, row: Mapping[str, str] | None = None) -> tuple[int, int]:
    width = _optional_first_present(row or {}, "width", "image_width", "Columns")
    height = _optional_first_present(row or {}, "height", "image_height", "Rows")
    if width is not None and height is not None:
        return int(float(width)), int(float(height))
    if not path.exists():
        raise VinDrDatasetUnavailableError(
            f"Cannot infer VinDr image dimensions; file missing: {path}"
        )
    if path.suffix.lower() in {".dcm", ".dicom"}:
        try:
            import pydicom

            metadata = pydicom.dcmread(path, stop_before_pixels=True)
            return int(metadata.Columns), int(metadata.Rows)
        except Exception as exc:  # pragma: no cover - depends on local DICOM files.
            raise VinDrDatasetUnavailableError(f"Failed to read DICOM metadata: {path}") from exc
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
            raise VinDrDatasetUnavailableError(f"Failed to read DICOM image: {path}") from exc
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
        raise ValueError(f"Invalid normalized bbox after clipping: {bbox}")
    return x_min, y_min, x_max, y_max


def _resolve_under_root(root: Path, path: str | Path | None) -> Path:
    if path is None:
        raise VinDrDatasetUnavailableError("Required VinDr path is missing from config.")
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


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
