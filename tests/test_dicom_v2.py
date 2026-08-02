"""Synthetic DICOM preprocessing contract tests."""

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from medguard.data.dicom import preprocess_dicom


def test_dicom_rescale_inversion_and_channel_contract(tmp_path: Path) -> None:
    mono2 = _write_dicom(tmp_path / "mono2.dcm", "MONOCHROME2")
    mono1 = _write_dicom(tmp_path / "mono1.dcm", "MONOCHROME1")

    normal = preprocess_dicom(mono2, channels=1, lower_percentile=0, upper_percentile=100)
    inverted = preprocess_dicom(mono1, channels=3, lower_percentile=0, upper_percentile=100)

    assert normal.image.shape == (1, 2, 3)
    assert inverted.image.shape == (3, 2, 3)
    assert normal.image.dtype == np.float32
    assert np.isfinite(normal.image).all()
    assert np.allclose(inverted.image[0], 1.0 - normal.image[0])
    assert inverted.provenance["monochrome1_inverted"] is True
    assert inverted.provenance["rescale_slope"] == 2.0
    assert set(inverted.provenance).isdisjoint({"PatientName", "PatientID"})


def _write_dicom(path: Path, photometric: str) -> Path:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = "DX"
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = 2
    dataset.RescaleIntercept = -10
    dataset.PatientName = "NOT^EXPORTED"
    dataset.PatientID = "private-patient"
    dataset.PixelData = np.arange(6, dtype=np.uint16).reshape(2, 3).tobytes()
    dataset.save_as(path, write_like_original=False)
    return path
