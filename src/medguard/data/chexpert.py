"""CheXpert access preflight and uncertainty-label policy primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from medguard.data.base import (
    DatasetAccessReport,
    DatasetManifest,
    SplitManifest,
    ValidationReport,
)
from medguard.data.manifests import build_patient_split, validate_manifest

UncertaintyPolicy = Literal["u_zero", "u_one", "u_ignore"]


def apply_uncertainty_policy(
    labels: torch.Tensor,
    policy: UncertaintyPolicy,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert CheXpert -1 labels and return a loss mask selected on validation."""

    uncertain = labels == -1
    targets = labels.clone().to(dtype=torch.float32)
    mask = torch.ones_like(targets, dtype=torch.bool)
    if policy == "u_zero":
        targets[uncertain] = 0
    elif policy == "u_one":
        targets[uncertain] = 1
    elif policy == "u_ignore":
        targets[uncertain] = 0
        mask[uncertain] = False
    else:
        raise ValueError(f"Unsupported CheXpert uncertainty policy: {policy}")
    return targets, mask


class CheXpertAdapter:
    """Restricted CheXpert adapter foundation; row parsing follows after access preflight."""

    name = "chexpert"
    restricted = True

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def check_access(self) -> DatasetAccessReport:
        train_csv = self.root / "train.csv"
        valid_csv = self.root / "valid.csv"
        images = self.root.exists() and any(self.root.glob("**/*.jpg"))
        accessible = train_csv.exists() and valid_csv.exists() and images
        return DatasetAccessReport(
            dataset=self.name,
            status="pending" if accessible else "blocked_external_access",
            metadata=train_csv.exists() and valid_csv.exists(),
            images=images,
            labels=train_csv.exists(),
            restricted=True,
            required_action="" if accessible else "Accept CheXpert terms and place data in Drive.",
        )

    def build_manifest(self) -> DatasetManifest:
        raise NotImplementedError("CheXpert manifest parsing requires an accessible licensed copy")

    def validate_manifest(self, manifest: DatasetManifest) -> ValidationReport:
        return validate_manifest(manifest)

    def build_split(self, manifest: DatasetManifest, seed: int = 2026) -> SplitManifest:
        return build_patient_split(manifest, seed=seed)

    def get_classification_dataset(self, split: str) -> Dataset[Any]:
        raise NotImplementedError("CheXpert tensor dataset construction is not in this first slice")

    def get_detection_dataset(self, split: str) -> None:
        return None
