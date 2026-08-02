"""Common dataset contracts for the Academic V2 research pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from torch.utils.data import Dataset

DatasetStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "blocked_external_access",
    "skipped_by_selection",
    "skipped_by_design",
]


@dataclass(frozen=True)
class DatasetAccessReport:
    """Preflight result that never contains credentials."""

    dataset: str
    status: DatasetStatus
    metadata: bool
    images: bool
    labels: bool
    reports: bool = False
    boxes: bool = False
    restricted: bool = False
    required_action: str = ""

    @property
    def accessible(self) -> bool:
        return self.metadata and self.images and self.labels


@dataclass(frozen=True)
class ManifestRow:
    """Minimum normalized record shared by every CXR dataset."""

    dataset: str
    patient_id: str
    study_id: str
    image_id: str
    split: str
    view: str
    image_path: str
    labels: dict[str, float | None]
    uncertainty_flags: dict[str, bool] = field(default_factory=dict)
    has_boxes: bool = False
    box_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetManifest:
    """Normalized manifest plus its privacy classification."""

    dataset: str
    rows: tuple[ManifestRow, ...]
    restricted: bool
    source: str = ""
    schema_version: str = "academic_v2.0"


@dataclass(frozen=True)
class ValidationReport:
    """Manifest validation and patient-leakage findings."""

    valid: bool
    row_count: int
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitManifest:
    """Patient-disjoint split assignment for one dataset manifest."""

    dataset: str
    rows: tuple[ManifestRow, ...]
    seed: int
    method: str = "patient_level_deterministic"


@runtime_checkable
class CXRDatasetAdapter(Protocol):
    """Interface implemented by all Academic V2 dataset adapters."""

    name: str
    restricted: bool

    def check_access(self) -> DatasetAccessReport: ...

    def build_manifest(self) -> DatasetManifest: ...

    def validate_manifest(self, manifest: DatasetManifest) -> ValidationReport: ...

    def build_split(self, manifest: DatasetManifest, seed: int = 2026) -> SplitManifest: ...

    def get_classification_dataset(self, split: str) -> Dataset[Any]: ...

    def get_detection_dataset(self, split: str) -> Dataset[Any] | None: ...


def resolve_data_root(workspace: str | Path, dataset: str) -> Path:
    """Return the canonical private workspace location for one dataset."""

    return Path(workspace).expanduser().resolve() / "datasets" / dataset
