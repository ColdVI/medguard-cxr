"""Dataset adapter registry with explicit access and privacy metadata."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from medguard.data.base import (
    CXRDatasetAdapter,
    DatasetAccessReport,
    DatasetManifest,
    SplitManifest,
    ValidationReport,
)
from medguard.data.manifests import build_patient_split, validate_manifest


@dataclass(frozen=True)
class DatasetRegistration:
    """Lazy adapter registration to avoid imports with optional dependencies."""

    name: str
    factory: Callable[..., CXRDatasetAdapter]
    restricted: bool
    tasks: tuple[str, ...]


class DatasetRegistry:
    """Case-insensitive registry for Academic V2 datasets."""

    def __init__(self) -> None:
        self._entries: dict[str, DatasetRegistration] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., CXRDatasetAdapter],
        *,
        restricted: bool,
        tasks: tuple[str, ...],
    ) -> None:
        key = name.lower()
        if key in self._entries:
            raise ValueError(f"Dataset adapter already registered: {name}")
        self._entries[key] = DatasetRegistration(key, factory, restricted, tasks)

    def create(self, name: str, **kwargs: object) -> CXRDatasetAdapter:
        """Instantiate one adapter by stable dataset identifier."""

        key = name.lower()
        if key not in self._entries:
            raise KeyError(f"Unknown dataset adapter {name!r}; available={self.names()}")
        return self._entries[key].factory(**kwargs)

    def registration(self, name: str) -> DatasetRegistration:
        return self._entries[name.lower()]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))


DATASETS = DatasetRegistry()


class FilesystemPreflightAdapter:
    """Common contract for datasets whose v2 row parser is still access-gated."""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str,
        restricted: bool,
        metadata_patterns: tuple[str, ...],
        image_patterns: tuple[str, ...] = ("**/*.png", "**/*.jpg", "**/*.dcm"),
        boxes: bool = False,
    ) -> None:
        self.root = Path(root)
        self.name = name
        self.restricted = restricted
        self.metadata_patterns = metadata_patterns
        self.image_patterns = image_patterns
        self.has_box_labels = boxes

    def check_access(self) -> DatasetAccessReport:
        metadata = all(any(self.root.glob(pattern)) for pattern in self.metadata_patterns)
        images = any(any(self.root.glob(pattern)) for pattern in self.image_patterns)
        accessible = metadata and images
        return DatasetAccessReport(
            dataset=self.name,
            status="pending" if accessible else "blocked_external_access",
            metadata=metadata,
            images=images,
            labels=metadata,
            boxes=self.has_box_labels and metadata,
            restricted=self.restricted,
            required_action="" if accessible else f"Provide authorized {self.name} data in Drive.",
        )

    def build_manifest(self) -> DatasetManifest:
        raise NotImplementedError(
            f"{self.name} Academic V2 manifest parser is not in the initial registry slice"
        )

    def validate_manifest(self, manifest: DatasetManifest) -> ValidationReport:
        return validate_manifest(manifest)

    def build_split(self, manifest: DatasetManifest, seed: int = 2026) -> SplitManifest:
        return build_patient_split(manifest, seed=seed)

    def get_classification_dataset(self, split: str) -> Dataset[Any]:
        raise NotImplementedError(
            f"{self.name} v2 tensor adapter is unavailable until manifest parsing completes"
        )

    def get_detection_dataset(self, split: str) -> Dataset[Any] | None:
        return None


def register_builtin_adapters() -> DatasetRegistry:
    """Load built-ins exactly once; imports stay local to keep startup light."""

    if DATASETS.names():
        return DATASETS
    from medguard.data.chexpert import CheXpertAdapter
    from medguard.data.mimic import MIMICCXRJPGAdapter

    DATASETS.register(
        "nih",
        lambda root: FilesystemPreflightAdapter(
            root,
            name="nih",
            restricted=False,
            metadata_patterns=("Data_Entry_2017.csv",),
        ),
        restricted=False,
        tasks=("classification", "calibration"),
    )
    DATASETS.register(
        "chexpert",
        CheXpertAdapter,
        restricted=True,
        tasks=("classification",),
    )
    DATASETS.register(
        "mimic",
        MIMICCXRJPGAdapter,
        restricted=True,
        tasks=("classification", "vlm"),
    )
    DATASETS.register(
        "vindr",
        lambda root: FilesystemPreflightAdapter(
            root,
            name="vindr",
            restricted=True,
            metadata_patterns=("**/*annotation*.csv",),
            boxes=True,
        ),
        restricted=True,
        tasks=("classification", "detection"),
    )
    DATASETS.register(
        "rsna",
        lambda root: FilesystemPreflightAdapter(
            root,
            name="rsna",
            restricted=True,
            metadata_patterns=("stage_2_train_labels.csv",),
            boxes=True,
        ),
        restricted=True,
        tasks=("weak_localization", "detection"),
    )
    return DATASETS
