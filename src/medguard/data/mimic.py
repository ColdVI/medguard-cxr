"""MIMIC-CXR-JPG restricted-data access adapter foundation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from medguard.data.base import (
    DatasetAccessReport,
    DatasetManifest,
    SplitManifest,
    ValidationReport,
)
from medguard.data.manifests import build_patient_split, validate_manifest


class MIMICCXRJPGAdapter:
    """Privacy-aware adapter that blocks honestly until credentialed data exists."""

    name = "mimic"
    restricted = True

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def check_access(self) -> DatasetAccessReport:
        metadata = next(iter(self.root.glob("*metadata*.csv*")), None)
        labels = next(iter(self.root.glob("*chexpert*.csv*")), None)
        reports = (self.root / "files").exists() and any(self.root.glob("**/*.txt"))
        images = (self.root / "files").exists() and any(self.root.glob("**/*.jpg"))
        accessible = metadata is not None and labels is not None and images
        return DatasetAccessReport(
            dataset=self.name,
            status="pending" if accessible else "blocked_external_access",
            metadata=metadata is not None,
            images=images,
            labels=labels is not None,
            reports=reports,
            restricted=True,
            required_action=(
                "" if accessible else "Complete PhysioNet credentialing, CITI training, and DUA."
            ),
        )

    def build_manifest(self) -> DatasetManifest:
        raise NotImplementedError("MIMIC manifest parsing requires credentialed local data")

    def validate_manifest(self, manifest: DatasetManifest) -> ValidationReport:
        return validate_manifest(manifest)

    def build_split(self, manifest: DatasetManifest, seed: int = 2026) -> SplitManifest:
        return build_patient_split(manifest, seed=seed)

    def get_classification_dataset(self, split: str) -> Dataset[Any]:
        raise NotImplementedError("MIMIC tensor dataset construction is not in this first slice")

    def get_detection_dataset(self, split: str) -> None:
        return None
