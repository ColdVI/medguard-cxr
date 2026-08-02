"""Patient-safety and restricted-manifest contract tests."""

from pathlib import Path

import pytest

from medguard.data.base import DatasetManifest, ManifestRow
from medguard.data.manifests import (
    RestrictedManifestWriteError,
    build_patient_split,
    validate_manifest,
    write_manifest,
)
from medguard.data.registry import register_builtin_adapters


def _row(patient: str, image: str, split: str = "train") -> ManifestRow:
    return ManifestRow(
        dataset="fixture",
        patient_id=patient,
        study_id=f"study-{image}",
        image_id=image,
        split=split,
        view="PA",
        image_path=f"/private/{image}.png",
        labels={"Atelectasis": 0.0},
    )


def test_patient_overlap_fails_before_training() -> None:
    manifest = DatasetManifest(
        dataset="fixture",
        rows=(_row("p1", "a", "train"), _row("p1", "b", "test")),
        restricted=False,
    )

    report = validate_manifest(manifest)

    assert not report.valid
    assert any("patient overlap" in error for error in report.errors)


def test_patient_split_is_deterministic_and_disjoint() -> None:
    manifest = DatasetManifest(
        dataset="fixture",
        rows=tuple(_row(f"p{i // 2}", f"image-{i}") for i in range(20)),
        restricted=False,
    )

    first = build_patient_split(manifest, seed=2026, val_fraction=0.2, test_fraction=0.2)
    second = build_patient_split(manifest, seed=2026, val_fraction=0.2, test_fraction=0.2)

    assert first == second
    assert validate_manifest(
        DatasetManifest(dataset="fixture", rows=first.rows, restricted=False)
    ).valid
    by_patient: dict[str, set[str]] = {}
    for row in first.rows:
        by_patient.setdefault(row.patient_id, set()).add(row.split)
    assert all(len(splits) == 1 for splits in by_patient.values())


def test_restricted_manifest_cannot_leave_private_root(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        dataset="fixture",
        rows=(_row("p1", "a"),),
        restricted=True,
    )
    private = tmp_path / "drive-private"
    private.mkdir()

    with pytest.raises(RestrictedManifestWriteError):
        write_manifest(manifest, tmp_path / "public" / "manifest.jsonl", private_root=private)

    target = write_manifest(
        manifest,
        private / "cache" / "manifest.jsonl",
        private_root=private,
    )
    assert target.is_file()


def test_dataset_registry_contains_all_mandatory_sources(tmp_path: Path) -> None:
    registry = register_builtin_adapters()

    assert set(registry.names()) == {"nih", "chexpert", "mimic", "vindr", "rsna"}
    report = registry.create("mimic", root=tmp_path / "missing").check_access()
    assert report.status == "blocked_external_access"
    assert report.restricted is True
