"""NIH ChestX-ray14 dataset and dataloader helpers."""

from __future__ import annotations

import csv
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from medguard.data.transforms import build_image_transform

PHASE = "1"

NIH_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


def is_available() -> bool:
    """Return whether the Phase 1 NIH dataset implementation is available."""
    return True


class DatasetUnavailableError(RuntimeError):
    """Raised when the local NIH dataset files are not present."""


@dataclass(frozen=True)
class NIHRecord:
    """One NIH image record."""

    image_id: str
    findings: str
    patient_id: str
    label: torch.Tensor
    path: Path


class NIHChestXray14Dataset(Dataset[dict[str, Any]]):
    """NIH ChestX-ray14 multi-label dataset.

    The official ``train_val_list.txt`` and ``test_list.txt`` files are honored
    when present. The train/validation split is then made at patient level from
    the official train-val pool using the configured seed and validation fraction.
    """

    def __init__(
        self,
        root: str | Path,
        metadata_csv: str | Path,
        split: str,
        labels: list[str],
        transform: Any,
        train_val_list: str | Path | None = None,
        test_list: str | Path | None = None,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 2026,
    ) -> None:
        self.root = Path(root)
        self.metadata_csv = _resolve_under_root(self.root, metadata_csv)
        self.split = split
        self.labels = labels
        self.transform = transform
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if not self.metadata_csv.exists():
            raise DatasetUnavailableError(f"Missing NIH metadata CSV: {self.metadata_csv}")

        train_val_ids = _read_split_ids(_resolve_optional_under_root(self.root, train_val_list))
        test_ids = _read_split_ids(_resolve_optional_under_root(self.root, test_list))
        all_records = self._read_metadata()
        _assert_official_patient_disjoint(all_records, train_val_ids, test_ids)
        self.records = self._select_split(all_records, train_val_ids, test_ids)
        if not self.records:
            raise DatasetUnavailableError(f"No NIH records found for split '{split}'.")

    @classmethod
    def from_config(cls, config: Mapping[str, Any], split: str) -> NIHChestXray14Dataset:
        """Create a dataset from the YAML configuration mapping."""
        data_cfg = config.get("data", {})
        split_cfg = config.get("split", {})
        return cls(
            root=data_cfg.get("root", "data/nih"),
            metadata_csv=data_cfg.get("image_index_csv", "Data_Entry_2017.csv"),
            split=split,
            labels=list(data_cfg.get("labels", NIH_LABELS)),
            transform=build_image_transform(config, train=split == "train"),
            train_val_list=data_cfg.get("train_val_list"),
            test_list=data_cfg.get("test_list"),
            val_fraction=float(split_cfg.get("val_fraction", 0.1)),
            test_fraction=float(split_cfg.get("test_fraction", 0.1)),
            seed=int(config.get("seed", 2026)),
        )

    def __len__(self) -> int:
        """Return split size."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one transformed sample."""
        record = self.records[index]
        path = self._resolve_image_path(record.path, record.image_id)
        image = self.transform(path)
        return {
            "image": image,
            "label": record.label.clone(),
            "patient_id": record.patient_id,
            "path": str(path),
        }

    def labels_tensor(self) -> torch.Tensor:
        """Return all labels as ``[N, C]`` tensor."""
        return torch.stack([record.label for record in self.records])

    def _read_metadata(self) -> list[NIHRecord]:
        records: list[NIHRecord] = []
        with self.metadata_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = _first_present(row, "Image Index", "image_id", "ImageID")
                findings = _first_present(
                    row,
                    "Finding Labels",
                    "finding_labels",
                    default="No Finding",
                )
                patient_id = _first_present(row, "Patient ID", "patient_id", default=image_id)
                label = _label_vector(findings, self.labels)
                records.append(
                    NIHRecord(
                        image_id=image_id,
                        findings=findings,
                        patient_id=str(patient_id),
                        label=label,
                        path=self.root / "images" / image_id,
                    )
                )
        return records

    def _select_split(
        self,
        records: list[NIHRecord],
        train_val_ids: set[str],
        test_ids: set[str],
    ) -> list[NIHRecord]:
        if train_val_ids or test_ids:
            train_val_records = [record for record in records if record.image_id in train_val_ids]
            test_records = [record for record in records if record.image_id in test_ids]
            if not train_val_records and train_val_ids:
                raise DatasetUnavailableError(
                    "Official train/val split file matched no metadata rows."
                )
            if self.split == "test":
                return test_records
            if test_ids and not train_val_ids:
                test_patients = {record.patient_id for record in test_records}
                non_test_records = [
                    record for record in records if record.patient_id not in test_patients
                ]
            else:
                non_test_records = [record for record in records if record.image_id not in test_ids]
            return _patient_train_val_split(
                train_val_records or non_test_records,
                split=self.split,
                val_fraction=self.val_fraction,
                seed=self.seed,
            )

        return _patient_train_val_test_split(
            records,
            split=self.split,
            val_fraction=self.val_fraction,
            test_fraction=self.test_fraction,
            seed=self.seed,
        )

    def _resolve_image_path(self, configured_path: Path, image_id: str) -> Path:
        candidates = [
            configured_path,
            self.root / image_id,
            self.root / "images" / image_id,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        matches = list(self.root.glob(f"**/{image_id}"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not find NIH image '{image_id}' under {self.root}.")


def dataset_available(config: Mapping[str, Any]) -> bool:
    """Return whether the configured NIH metadata exists locally."""
    data_cfg = config.get("data", {})
    root = Path(data_cfg.get("root", "data/nih"))
    metadata = _resolve_under_root(root, data_cfg.get("image_index_csv", "Data_Entry_2017.csv"))
    return metadata.exists()


def create_dataloader(
    dataset: Dataset[dict[str, Any]],
    config: Mapping[str, Any],
    shuffle: bool,
    profile: str | None = None,
    batch_size: int | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create a DataLoader using only YAML-provided loader settings."""
    loader_kwargs = dataloader_kwargs(config, profile=profile)
    resolved_batch_size = int(batch_size or config.get("training", {}).get("batch_size", 32))
    return DataLoader(dataset, batch_size=resolved_batch_size, shuffle=shuffle, **loader_kwargs)


def dataloader_kwargs(config: Mapping[str, Any], profile: str | None = None) -> dict[str, Any]:
    """Return PyTorch DataLoader kwargs from the selected YAML profile."""
    dataloader_cfg = config.get("dataloader", {})
    profiles = dataloader_cfg.get("profiles", {})
    profile_name = profile or dataloader_cfg.get("active_profile", "cpu_ci")
    if profile_name not in profiles:
        raise KeyError(f"Unknown dataloader profile: {profile_name}")

    selected = profiles[profile_name]
    num_workers = int(selected.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(selected.get("pin_memory", False)),
    }
    if num_workers > 0:
        if selected.get("prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(selected["prefetch_factor"])
        kwargs["persistent_workers"] = bool(selected.get("persistent_workers", False))
    return kwargs


def compute_pos_weight(dataset: NIHChestXray14Dataset) -> torch.Tensor:
    """Compute BCEWithLogitsLoss ``pos_weight`` as negative_count / positive_count."""
    labels = dataset.labels_tensor().to(dtype=torch.float32)
    positive = labels.sum(dim=0)
    negative = labels.shape[0] - positive
    weights = torch.where(
        positive > 0,
        negative / positive.clamp_min(1.0),
        torch.ones_like(positive),
    )
    return weights.to(dtype=torch.float32)


def _assert_official_patient_disjoint(
    records: list[NIHRecord],
    train_val_ids: set[str],
    test_ids: set[str],
) -> None:
    if not train_val_ids or not test_ids:
        return

    train_val_patients = {
        record.patient_id for record in records if record.image_id in train_val_ids
    }
    test_patients = {record.patient_id for record in records if record.image_id in test_ids}
    overlap = train_val_patients & test_patients
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise DatasetUnavailableError(
            "Official NIH split files contain patient overlap between train/val and test: "
            f"{examples}"
        )


def _label_vector(findings: str, labels: list[str]) -> torch.Tensor:
    finding_set = {finding.strip() for finding in findings.split("|") if finding.strip()}
    return torch.tensor(
        [1.0 if label in finding_set else 0.0 for label in labels],
        dtype=torch.float32,
    )


def _patient_train_val_split(
    records: list[NIHRecord],
    split: str,
    val_fraction: float,
    seed: int,
) -> list[NIHRecord]:
    patients = _shuffled_patients(records, seed)
    val_patients = set(patients[: _bounded_count(len(patients), val_fraction)])
    if split == "val":
        return [record for record in records if record.patient_id in val_patients]
    return [record for record in records if record.patient_id not in val_patients]


def _patient_train_val_test_split(
    records: list[NIHRecord],
    split: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> list[NIHRecord]:
    patients = _shuffled_patients(records, seed)
    test_count = _bounded_count(len(patients), test_fraction)
    remaining = patients[test_count:]
    val_count = _bounded_count(len(remaining), val_fraction)
    test_patients = set(patients[:test_count])
    val_patients = set(remaining[:val_count])

    if split == "test":
        return [record for record in records if record.patient_id in test_patients]
    if split == "val":
        return [record for record in records if record.patient_id in val_patients]
    return [
        record
        for record in records
        if record.patient_id not in test_patients and record.patient_id not in val_patients
    ]


def _shuffled_patients(records: list[NIHRecord], seed: int) -> list[str]:
    patients = sorted({record.patient_id for record in records})
    random.Random(seed).shuffle(patients)
    return patients


def _bounded_count(total: int, fraction: float) -> int:
    if total <= 1 or fraction <= 0:
        return 0
    return max(1, min(total - 1, round(total * fraction)))


def _read_split_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _resolve_under_root(root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


def _resolve_optional_under_root(root: Path, path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return _resolve_under_root(root, path)


def _first_present(row: Mapping[str, str], *names: str, default: str | None = None) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    if default is not None:
        return default
    raise KeyError(f"None of the required columns were present: {names}")
