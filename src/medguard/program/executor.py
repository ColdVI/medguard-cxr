"""First-slice executor: safe preflight and synthetic contract proof."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medguard.data.base import DatasetManifest, ManifestRow
from medguard.data.cache import atomic_write_text
from medguard.data.manifests import build_patient_split, manifest_hash, validate_manifest
from medguard.program.planner import RunSpec
from medguard.program.registry import RunRegistry
from medguard.program.state import RunStatus


def execute_contract_smoke(spec: RunSpec, registry: RunRegistry, resume: bool) -> dict[str, Any]:
    """Prove plumbing without emitting synthetic performance metrics or DONE.json."""

    status = registry.register(spec)
    run_dir = registry.run_directory(spec.run_id)
    proof_path = run_dir / "SMOKE_PROOF.json"
    if resume and proof_path.exists() and status["status"] == RunStatus.SKIPPED_BY_DESIGN.value:
        return json.loads(proof_path.read_text(encoding="utf-8"))
    if status["status"] != RunStatus.PENDING.value:
        raise ValueError(f"Cannot start synthetic proof from {status['status']}")
    registry.transition(spec.run_id, RunStatus.RUNNING, detail="validating synthetic contracts")
    rows = tuple(
        ManifestRow(
            dataset="synthetic",
            patient_id=f"patient-{index}",
            study_id=f"study-{index}",
            image_id=f"image-{index}",
            split="train",
            view="PA",
            image_path=f"synthetic://image-{index}",
            labels={"Atelectasis": float(index % 2)},
        )
        for index in range(24)
    )
    manifest = DatasetManifest(dataset="synthetic", rows=rows, restricted=False)
    split = build_patient_split(manifest, seed=spec.seed, val_fraction=0.2, test_fraction=0.2)
    split_manifest = DatasetManifest(dataset="synthetic", rows=split.rows, restricted=False)
    report = validate_manifest(split_manifest)
    if not report.valid:
        registry.transition(
            spec.run_id,
            RunStatus.FAILED,
            detail="synthetic contract validation failed",
            extra={"errors": list(report.errors)},
        )
        raise RuntimeError("Synthetic contract validation failed: " + "; ".join(report.errors))
    proof = {
        "artifact_kind": "smoke",
        "smoke_status": "passed",
        "research_status": "not_run",
        "performance_metrics_emitted": False,
        "patient_leakage": False,
        "row_count": report.row_count,
        "manifest_hash": manifest_hash(split_manifest),
        "warning": "Synthetic contract proof only; not evidence of model performance.",
    }
    atomic_write_text(proof_path, json.dumps(proof, indent=2, sort_keys=True) + "\n")
    registry.transition(
        spec.run_id,
        RunStatus.SKIPPED_BY_DESIGN,
        detail="synthetic contracts passed; research execution intentionally not claimed",
        extra={"smoke_status": "passed", "artifact_kind": "smoke"},
    )
    return proof


def register_research_plan(
    specs: list[RunSpec],
    registry: RunRegistry,
    workspace: Path,
) -> list[dict[str, Any]]:
    """Register real runs and block only datasets whose required roots are absent."""

    reports: list[dict[str, Any]] = []
    for spec in specs:
        status = registry.register(spec)
        if status["status"] != RunStatus.PENDING.value:
            reports.append(status)
            continue
        if spec.dataset in {"selection_dependent", "pooled", "legacy"}:
            reports.append(status)
            continue
        data_root = workspace / "datasets" / spec.dataset
        if not data_root.exists():
            status = registry.transition(
                spec.run_id,
                RunStatus.BLOCKED_EXTERNAL_ACCESS,
                detail=f"dataset root unavailable: {data_root}",
            )
        reports.append(status)
    return reports
