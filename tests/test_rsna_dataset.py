"""Phase 3 tests for RSNA Pneumonia localization fallback parsing."""

from pathlib import Path

import numpy as np
import pydicom
import yaml
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from medguard.data.rsna import (
    NIH_TO_RSNA_LABELS,
    RSNAPneumoniaDataset,
    dataset_available,
    normalize_rsna_bbox,
    read_rsna_image,
    rsna_labels_for_nih_label,
)


def test_normalize_rsna_bbox_xywh() -> None:
    """RSNA pixel xywh boxes are normalized into xyxy coordinates."""

    assert normalize_rsna_bbox(2, 1, 8, 5, image_width=20, image_height=10) == (
        0.1,
        0.1,
        0.5,
        0.6,
    )


def test_rsna_dataset_loads_positive_boxes_and_dimensions(tmp_path: Path) -> None:
    """The RSNA loader preserves image dimensions and normalized boxes."""

    root = _write_rsna_fixture(tmp_path)

    dataset = RSNAPneumoniaDataset(
        root=root,
        manifest_csv="manifest.csv",
        split="val",
    )

    assert len(dataset) == 1
    assert dataset.records[0].width == 20
    assert dataset.records[0].height == 10
    assert dataset.records[0].boxes[0].label == "Lung Opacity"
    assert dataset.records[0].boxes[0].bbox == (0.1, 0.1, 0.5, 0.6)
    sample = dataset[0]
    assert sample["image_id"] == "case_001"
    assert sample["label"].item() == 1.0
    assert sample["boxes"].shape == (1, 4)
    assert sample["box_labels"] == ["Lung Opacity"]
    assert sample["image"].shape == (1, 10, 20)


def test_rsna_dataset_preserves_multiple_boxes(tmp_path: Path) -> None:
    """Multiple positive rows for one patient stay as multiple GT boxes."""

    root = _write_rsna_fixture(
        tmp_path,
        labels=(
            "patientId,x,y,width,height,Target\n"
            "case_001,2,1,8,5,1\n"
            "case_001,4,2,6,3,1\n"
        ),
    )

    dataset = RSNAPneumoniaDataset(
        root=root,
        manifest_csv="manifest.csv",
        split="val",
    )

    assert len(dataset.records[0].boxes) == 2


def test_rsna_dataset_can_include_negative_rows_explicitly(tmp_path: Path) -> None:
    """Negative examples are included only when explicitly requested."""

    root = _write_rsna_fixture(
        tmp_path,
        labels=(
            "patientId,x,y,width,height,Target\n"
            "case_001,,,,,0\n"
        ),
    )

    dataset = RSNAPneumoniaDataset(
        root=root,
        manifest_csv="manifest.csv",
        split="val",
        include_negative=True,
    )

    assert dataset.records[0].boxes == ()
    assert dataset[0]["label"].item() == 0.0
    assert dataset[0]["boxes"].shape == (0, 4)


def test_rsna_loader_reads_dicom_dimensions_and_pixels(tmp_path: Path) -> None:
    """RSNA DICOM images are read as grayscale while preserving dimensions."""

    root = tmp_path / "rsna"
    image_dir = root / "stage_2_train_images"
    image_dir.mkdir(parents=True)
    _write_dicom(image_dir / "case_001.dcm", width=20, height=10)
    (root / "stage_2_train_labels.csv").write_text(
        "patientId,x,y,width,height,Target\ncase_001,2,1,8,5,1\n",
        encoding="utf-8",
    )
    (root / "stage_2_detailed_class_info.csv").write_text(
        "patientId,class\ncase_001,Lung Opacity\n",
        encoding="utf-8",
    )
    (root / "manifest.csv").write_text("patientId,split\ncase_001,val\n", encoding="utf-8")

    dataset = RSNAPneumoniaDataset(root=root, manifest_csv="manifest.csv", split="val")
    image = read_rsna_image(dataset.records[0].path)

    assert dataset.records[0].width == 20
    assert dataset.records[0].height == 10
    assert image.size == (20, 10)
    assert dataset[0]["image"].shape == (1, 10, 20)


def test_rsna_dataset_available_requires_csv_and_image(tmp_path: Path) -> None:
    """Availability checks require the labels CSV plus a resolvable image."""

    root = tmp_path / "rsna"
    root.mkdir()
    config = {"data": {"root": str(root), "labels_csv": "stage_2_train_labels.csv"}}
    assert dataset_available(config) is False

    image_dir = root / "stage_2_train_images"
    image_dir.mkdir()
    Image.new("L", (20, 10), color=128).save(image_dir / "case_001.png")
    (root / "stage_2_train_labels.csv").write_text(
        "patientId,x,y,width,height,Target\ncase_001,2,1,8,5,1\n",
        encoding="utf-8",
    )
    assert dataset_available(config) is True


def test_nih_to_rsna_mapping_is_explicit() -> None:
    """Only the coarse pneumonia-to-lung-opacity bridge is enabled."""

    assert rsna_labels_for_nih_label("Pneumonia") == ["Lung Opacity"]
    assert NIH_TO_RSNA_LABELS["Mass"] == []
    assert NIH_TO_RSNA_LABELS["Pneumothorax"] == []


def test_grounding_rsna_config_records_phase4a_go_and_phase4b_block() -> None:
    """The RSNA config records Phase 4A entry and Phase 4B real-checkpoint gate."""

    config = yaml.safe_load(Path("configs/grounding_rsna.yaml").read_text(encoding="utf-8"))
    gate = config["phase_gate"]

    assert (
        gate["phase4_entry_status"]
        == "phase4a_engineering_allowed_phase4b_blocked_until_real_checkpoint"
    )
    assert gate["require_real_localization_before_phase4a"] is False
    assert gate["require_real_checkpoint_before_phase4b"] is True
    assert gate["synthetic_smoke_is_blocking_for_phase4b"] is True
    assert "required_before_phase4b" in gate
    assert "rsna-pneumonia-detection" in gate["accepted_real_localization_datasets"]
    assert gate["accepted_real_localization_datasets"] == ["rsna-pneumonia-detection"]


def _write_rsna_fixture(
    tmp_path: Path,
    labels: str = "patientId,x,y,width,height,Target\ncase_001,2,1,8,5,1\n",
) -> Path:
    root = tmp_path / "rsna"
    image_dir = root / "stage_2_train_images"
    image_dir.mkdir(parents=True)
    Image.new("L", (20, 10), color=128).save(image_dir / "case_001.png")
    (root / "stage_2_train_labels.csv").write_text(labels, encoding="utf-8")
    (root / "stage_2_detailed_class_info.csv").write_text(
        "patientId,class\ncase_001,Lung Opacity\n",
        encoding="utf-8",
    )
    (root / "manifest.csv").write_text(
        "patientId,split\ncase_001,val\n",
        encoding="utf-8",
    )
    return root


def _write_dicom(path: Path, width: int, height: int) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "DX"
    dataset.Rows = height
    dataset.Columns = width
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.arange(width * height, dtype=np.uint8).reshape(height, width).tobytes()
    dataset.save_as(path, write_like_original=False)
