"""Phase 1 NIH dataloader tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image
from scripts.train_classifier import build_smoke_loaders

from medguard.data.nih import (
    DatasetUnavailableError,
    NIHChestXray14Dataset,
    compute_pos_weight,
    create_dataloader,
    dataset_available,
)


def test_nih_dataset_sample_shape_dtype_and_labels(tmp_path: Path) -> None:
    """Dataset returns image, multi-label target, patient_id, and path."""
    config = make_tiny_nih_config(tmp_path)
    dataset = NIHChestXray14Dataset.from_config(config, split="train")

    sample = dataset[0]

    assert sample["image"].shape == (3, 32, 32)
    assert str(sample["image"].dtype) == "torch.float32"
    assert sample["label"].shape == (14,)
    assert str(sample["label"].dtype) == "torch.float32"
    assert set(sample["label"].tolist()) <= {0.0, 1.0}
    assert sample["patient_id"]
    assert sample["path"].endswith(".png")


def test_official_split_files_and_dataloader_profile_are_used(tmp_path: Path) -> None:
    """Official split files are honored and DataLoader kwargs come from config."""
    config = make_tiny_nih_config(tmp_path)

    train_dataset = NIHChestXray14Dataset.from_config(config, split="train")
    val_dataset = NIHChestXray14Dataset.from_config(config, split="val")
    test_dataset = NIHChestXray14Dataset.from_config(config, split="test")
    loader = create_dataloader(train_dataset, config, shuffle=False)

    assert len(train_dataset) == 2
    assert len(val_dataset) == 1
    assert len(test_dataset) == 1
    assert loader.num_workers == 0
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 32, 32)
    assert batch["label"].shape == (2, 14)
    assert not patient_ids(train_dataset) & patient_ids(val_dataset)
    assert not patient_ids(train_dataset) & patient_ids(test_dataset)
    assert not patient_ids(val_dataset) & patient_ids(test_dataset)


def test_pos_weight_is_computed_from_training_labels(tmp_path: Path) -> None:
    """Temporary Phase 1 pos_weight fallback uses negative_count / positive_count."""

    class FakeDataset:
        def labels_tensor(self):
            import torch

            return torch.tensor(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            )

    pos_weight = compute_pos_weight(FakeDataset())

    assert pos_weight.tolist() == [1.0, 3.0]
    assert str(pos_weight.dtype) == "torch.float32"


def test_dataset_available_requires_referenced_images(tmp_path: Path) -> None:
    """Partial NIH downloads with metadata only fall back to smoke mode."""
    root = tmp_path / "partial_nih"
    root.mkdir()
    with (root / "Data_Entry_2017.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Image Index", "Finding Labels", "Patient ID"])
        writer.writerow(["missing.png", "Mass", "p1"])

    config = {
        "data": {
            "root": str(root),
            "image_index_csv": "Data_Entry_2017.csv",
        }
    }

    with pytest.warns(UserWarning, match="metadata exists"):
        assert dataset_available(config) is False


def test_smoke_loader_uses_configured_dataloader_profile(tmp_path: Path) -> None:
    """No-data smoke loaders still route through YAML DataLoader settings."""
    config = make_tiny_nih_config(tmp_path)
    config["smoke"] = {"samples": 4, "batch_size": 2, "image_size": 16}

    train_loader, val_loader, _ = build_smoke_loaders(config)

    assert train_loader.num_workers == 0
    assert val_loader.num_workers == 0


def test_official_split_patient_overlap_is_rejected(tmp_path: Path) -> None:
    """Malformed official split files cannot leak patients across train/test."""
    config = make_tiny_nih_config(tmp_path)
    root = Path(config["data"]["root"])
    with (root / "test_list.txt").open("a") as handle:
        handle.write("train_a.png\n")

    with pytest.raises(DatasetUnavailableError, match="patient overlap"):
        NIHChestXray14Dataset.from_config(config, split="train")


def patient_ids(dataset: NIHChestXray14Dataset) -> set[str]:
    """Return patient IDs for split-overlap assertions."""
    return {record.patient_id for record in dataset.records}


def make_tiny_nih_config(tmp_path: Path) -> dict:
    """Create a minimal NIH-like fixture and return config."""
    root = tmp_path / "nih"
    images = root / "images"
    images.mkdir(parents=True)

    rows = [
        ("train_a.png", "Atelectasis|Effusion", "p1"),
        ("train_b.png", "No Finding", "p2"),
        ("val_a.png", "Mass", "p3"),
        ("test_a.png", "Nodule", "p4"),
    ]
    for index, (image_id, _, _) in enumerate(rows):
        Image.new("L", (20, 24), color=40 + index * 20).save(images / image_id)

    with (root / "Data_Entry_2017.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Image Index", "Finding Labels", "Patient ID"])
        writer.writerows(rows)

    (root / "train_val_list.txt").write_text("train_a.png\ntrain_b.png\nval_a.png\n")
    (root / "test_list.txt").write_text("test_a.png\n")

    return {
        "seed": 7,
        "data": {
            "root": str(root),
            "image_index_csv": "Data_Entry_2017.csv",
            "train_val_list": "train_val_list.txt",
            "test_list": "test_list.txt",
            "labels": [
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
            ],
        },
        "split": {"val_fraction": 0.34, "test_fraction": 0.25},
        "preprocessing": {
            "image_size": 32,
            "channels": 3,
            "normalization": {
                "active": "imagenet",
                "imagenet_mean": [0.485, 0.456, 0.406],
                "imagenet_std": [0.229, 0.224, 0.225],
            },
            "train_augmentations": {
                "random_horizontal_flip": False,
                "random_horizontal_flip_probability": 0.0,
            },
        },
        "training": {"batch_size": 2},
        "dataloader": {
            "active_profile": "cpu_ci",
            "profiles": {
                "cpu_ci": {
                    "num_workers": 0,
                    "pin_memory": False,
                    "prefetch_factor": None,
                    "persistent_workers": False,
                }
            },
        },
    }
