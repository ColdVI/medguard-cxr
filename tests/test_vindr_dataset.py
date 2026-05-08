"""Phase 3 tests for VinDr-CXR dataset parsing."""

from pathlib import Path

import pytest
from PIL import Image

from medguard.data.vindr import (
    NIH_TO_VINDR_LABELS,
    VinDrBox,
    VinDrCXRDataset,
    VinDrDatasetUnavailableError,
    consensus_box_from_annotations,
    dataset_available,
    normalize_bbox,
    vindr_labels_for_nih_label,
)


def test_normalize_bbox_pixel_xyxy() -> None:
    """Pixel boxes are normalized against original dimensions."""

    assert normalize_bbox((2, 1, 10, 6), width=20, height=10) == (0.1, 0.1, 0.5, 0.6)


def test_vindr_dataset_loads_annotations_and_dimensions(tmp_path: Path) -> None:
    """The loader preserves dimensions and emits normalized boxes."""

    root = tmp_path / "vindr"
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    Image.new("L", (20, 10), color=128).save(image_dir / "case_001.png")
    (root / "annotations.csv").write_text(
        "image_id,class_name,x_min,y_min,x_max,y_max,split,path\n"
        "case_001,Nodule/Mass,2,1,10,6,test,images/case_001.png\n",
        encoding="utf-8",
    )

    dataset = VinDrCXRDataset(root=root, annotations_csv="annotations.csv", split="test")

    assert len(dataset) == 1
    assert dataset.records[0].width == 20
    assert dataset.records[0].height == 10
    assert dataset.records[0].boxes[0].bbox == (0.1, 0.1, 0.5, 0.6)
    sample = dataset[0]
    assert sample["image_id"] == "case_001"
    assert sample["boxes"].shape == (1, 4)
    assert sample["image"].shape == (1, 10, 20)


def test_vindr_dataset_merges_same_class_annotations_to_consensus(tmp_path: Path) -> None:
    """Radiologist boxes for the same image/class are merged by consensus."""

    root = tmp_path / "vindr"
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    Image.new("L", (20, 10), color=128).save(image_dir / "case_001.png")
    (root / "annotations.csv").write_text(
        "image_id,class_name,x_min,y_min,x_max,y_max,rad_id,split,path\n"
        "case_001,Nodule/Mass,2,1,10,6,r1,test,images/case_001.png\n"
        "case_001,Nodule/Mass,3,1,11,6,r2,test,images/case_001.png\n"
        "case_001,Nodule/Mass,2,2,10,7,r3,test,images/case_001.png\n",
        encoding="utf-8",
    )

    dataset = VinDrCXRDataset(root=root, annotations_csv="annotations.csv", split="test")

    assert len(dataset.records[0].boxes) == 1
    assert dataset.records[0].boxes[0].annotator_id == "consensus:iou_merge:n=3"
    assert dataset.records[0].boxes[0].bbox == (0.1, 0.1, 0.55, 0.7)


def test_consensus_majority_vote_returns_overlap_region() -> None:
    """Majority-vote consensus keeps cells where most annotators agree."""

    boxes = [
        VinDrBox(label="Cardiomegaly", bbox=(0.1, 0.1, 0.6, 0.6), annotator_id="r1"),
        VinDrBox(label="Cardiomegaly", bbox=(0.2, 0.2, 0.7, 0.7), annotator_id="r2"),
        VinDrBox(label="Cardiomegaly", bbox=(0.2, 0.2, 0.6, 0.6), annotator_id="r3"),
    ]

    consensus = consensus_box_from_annotations(
        boxes,
        strategy="majority_vote",
        min_annotators=2,
        grid_size=10,
    )

    assert consensus is not None
    assert consensus.bbox == (0.2, 0.2, 0.6, 0.6)


def test_consensus_failure_count_is_recorded(tmp_path: Path) -> None:
    """Noisy same-class boxes that fail consensus are counted for operators."""

    root = tmp_path / "vindr"
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    Image.new("L", (20, 10), color=128).save(image_dir / "case_001.png")
    (root / "manifest.csv").write_text(
        "image_id,patient_id,split,path\ncase_001,p1,test,images/case_001.png\n",
        encoding="utf-8",
    )
    (root / "annotations.csv").write_text(
        "image_id,class_name,x_min,y_min,x_max,y_max,rad_id,split,path\n"
        "case_001,Nodule/Mass,0,0,4,4,r1,test,images/case_001.png\n"
        "case_001,Nodule/Mass,12,5,18,9,r2,test,images/case_001.png\n",
        encoding="utf-8",
    )

    dataset = VinDrCXRDataset(
        root=root,
        annotations_csv="annotations.csv",
        manifest_csv="manifest.csv",
        split="test",
        consensus_min_annotators=2,
        allow_empty_annotations=True,
    )

    assert dataset.consensus_failure_count == 1
    assert dataset.consensus_failure_by_image == {"case_001": 1}
    assert dataset.records[0].boxes == ()


def test_nih_to_vindr_mapping_is_explicit() -> None:
    """Cross-dataset label mapping is explicit, including unmapped labels."""

    assert vindr_labels_for_nih_label("Mass") == ["Nodule/Mass"]
    assert vindr_labels_for_nih_label("Nodule") == ["Nodule/Mass"]
    assert vindr_labels_for_nih_label("Pleural_Thickening") == ["Pleural thickening"]
    assert NIH_TO_VINDR_LABELS["Pneumonia"] == []
    assert NIH_TO_VINDR_LABELS["Hernia"] == []


def test_vindr_dataset_does_not_silently_skip_missing_annotations(tmp_path: Path) -> None:
    """Manifest images without boxes raise unless explicitly allowed."""

    root = tmp_path / "vindr"
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    Image.new("L", (20, 10), color=128).save(image_dir / "case_001.png")
    (root / "manifest.csv").write_text(
        "image_id,patient_id,split,path\ncase_001,p1,test,images/case_001.png\n",
        encoding="utf-8",
    )
    (root / "annotations.csv").write_text(
        "image_id,class_name,x_min,y_min,x_max,y_max,split,path\n",
        encoding="utf-8",
    )

    with pytest.raises(VinDrDatasetUnavailableError, match="Missing VinDr annotations"):
        VinDrCXRDataset(
            root=root,
            annotations_csv="annotations.csv",
            manifest_csv="manifest.csv",
            split="test",
        )


def test_vindr_dataset_available_requires_image_and_annotation(tmp_path: Path) -> None:
    """Availability checks require CSV and at least one resolvable image."""

    root = tmp_path / "vindr"
    root.mkdir()
    config = {"data": {"root": str(root), "annotations_csv": "annotations.csv"}}
    assert dataset_available(config) is False

    image_dir = root / "images"
    image_dir.mkdir()
    Image.new("L", (20, 10), color=128).save(image_dir / "case_001.png")
    (root / "annotations.csv").write_text(
        "image_id,class_name,x_min,y_min,x_max,y_max,split,path\n"
        "case_001,Nodule/Mass,2,1,10,6,test,images/case_001.png\n",
        encoding="utf-8",
    )
    assert dataset_available(config) is True
