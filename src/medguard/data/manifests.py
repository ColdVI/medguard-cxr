"""Manifest hashing, validation, privacy guards, and patient-level splitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from medguard.data.base import DatasetManifest, ManifestRow, SplitManifest, ValidationReport

ALLOWED_SPLITS = {"train", "val", "test"}
FRONTAL_VIEWS = {"PA", "AP", "AP PORTABLE", "AP_PORTABLE"}


class RestrictedManifestWriteError(PermissionError):
    """Raised when private row-level metadata targets a public destination."""


def validate_manifest(manifest: DatasetManifest) -> ValidationReport:
    """Validate required fields and assert patient-disjoint split assignment."""

    errors: list[str] = []
    warnings: list[str] = []
    seen_images: set[str] = set()
    patient_splits: dict[str, set[str]] = {}
    for index, row in enumerate(manifest.rows):
        prefix = f"row {index}"
        if row.dataset != manifest.dataset:
            errors.append(f"{prefix}: dataset mismatch")
        for field_name in ("patient_id", "study_id", "image_id", "image_path"):
            if not getattr(row, field_name):
                errors.append(f"{prefix}: empty {field_name}")
        if row.split not in ALLOWED_SPLITS:
            errors.append(f"{prefix}: invalid split {row.split!r}")
        if row.image_id in seen_images:
            errors.append(f"{prefix}: duplicate image_id {row.image_id!r}")
        seen_images.add(row.image_id)
        patient_splits.setdefault(row.patient_id, set()).add(row.split)
        if row.box_count < 0 or row.has_boxes != (row.box_count > 0):
            errors.append(f"{prefix}: inconsistent box metadata")
        normalized_view = row.view.strip().upper()
        if normalized_view and normalized_view not in FRONTAL_VIEWS:
            warnings.append(f"{prefix}: non-primary view {row.view!r}")
    leaked = sorted(patient for patient, splits in patient_splits.items() if len(splits) > 1)
    if leaked:
        preview = ", ".join(leaked[:5])
        errors.append(f"patient overlap across splits: {preview}")
    return ValidationReport(
        valid=not errors,
        row_count=len(manifest.rows),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def assert_no_patient_leakage(manifest: DatasetManifest) -> None:
    """Stop a run before training if a patient occurs in multiple splits."""

    report = validate_manifest(manifest)
    leakage = [error for error in report.errors if error.startswith("patient overlap")]
    if leakage:
        raise ValueError(leakage[0])


def build_patient_split(
    manifest: DatasetManifest,
    seed: int = 2026,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> SplitManifest:
    """Assign whole patients deterministically without consulting test performance."""

    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("Split fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be below 1")
    assignment: dict[str, str] = {}
    rows: list[ManifestRow] = []
    for row in manifest.rows:
        split = assignment.setdefault(
            row.patient_id,
            _split_for_patient(row.patient_id, seed, val_fraction, test_fraction),
        )
        rows.append(replace(row, split=split))
    result = SplitManifest(dataset=manifest.dataset, rows=tuple(rows), seed=seed)
    validated = DatasetManifest(
        dataset=manifest.dataset,
        rows=result.rows,
        restricted=manifest.restricted,
        source=manifest.source,
        schema_version=manifest.schema_version,
    )
    assert_no_patient_leakage(validated)
    return result


def manifest_hash(manifest: DatasetManifest) -> str:
    """Create a stable hash used for run provenance and resume checks."""

    payload = {
        "dataset": manifest.dataset,
        "restricted": manifest.restricted,
        "schema_version": manifest.schema_version,
        "rows": [row.to_dict() for row in manifest.rows],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(
    manifest: DatasetManifest,
    path: str | Path,
    *,
    private_root: str | Path | None = None,
) -> Path:
    """Atomically write JSONL, rejecting restricted rows outside the private root."""

    target = Path(path).expanduser().resolve()
    if manifest.restricted:
        if private_root is None:
            raise RestrictedManifestWriteError(
                "Restricted manifests require an explicit private_root"
            )
        private = Path(private_root).expanduser().resolve()
        if not target.is_relative_to(private):
            raise RestrictedManifestWriteError(
                f"Restricted manifest target {target} is outside private root {private}"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    content = "\n".join(json.dumps(row.to_dict(), sort_keys=True) for row in manifest.rows)
    temporary.write_text(content + ("\n" if content else ""), encoding="utf-8")
    temporary.replace(target)
    return target


def _split_for_patient(
    patient_id: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> str:
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    if fraction < test_fraction:
        return "test"
    if fraction < test_fraction + val_fraction:
        return "val"
    return "train"
