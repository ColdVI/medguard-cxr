"""Checkpoint provenance and integrity primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from medguard.data.cache import file_sha256


@dataclass(frozen=True)
class CheckpointProvenance:
    model_id: str
    source: str
    source_revision: str
    license: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def inspect_checkpoint(
    path: str | Path,
    *,
    model_id: str,
    source: str,
    source_revision: str,
    license_name: str,
) -> CheckpointProvenance:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return CheckpointProvenance(
        model_id=model_id,
        source=source,
        source_revision=source_revision,
        license=license_name,
        sha256=file_sha256(checkpoint),
    )
