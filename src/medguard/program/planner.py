"""Deterministic, compute-aware program planning."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    task: str
    model: str
    loss: str
    resolution: int | str
    seed: int
    stage: str
    artifact_kind: str = "real"

    @property
    def run_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        suffix = hashlib.sha256(payload.encode()).hexdigest()[:10]
        parts = (
            self.dataset,
            self.task,
            self.model,
            self.loss,
            str(self.resolution),
            str(self.seed),
            suffix,
        )
        return "__".join(_slug(part) for part in parts)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "run_id": self.run_id}


def load_program(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        program = yaml.safe_load(handle)
    if program.get("schema_version") != "academic_v2.0":
        raise ValueError("Unsupported or missing Academic V2 program schema_version")
    if not isinstance(program.get("stages"), list):
        raise ValueError("Program config must contain a stages list")
    return program


def plan_runs(program: dict[str, Any], profile: str) -> list[RunSpec]:
    """Expand only explicitly enumerated stage grids, never a global Cartesian product."""

    if profile == "synthetic":
        return [
            RunSpec(
                dataset="synthetic",
                task="contract_smoke",
                model="none",
                loss="none",
                resolution=32,
                seed=2026,
                stage="S0",
                artifact_kind="smoke",
            )
        ]
    planned: list[RunSpec] = []
    for stage in program["stages"]:
        for grid in stage.get("explicit_grids", []):
            values = {
                "dataset": _as_list(grid.get("dataset", "selection_dependent")),
                "task": _as_list(grid.get("task", stage["task"])),
                "model": _as_list(grid.get("model", "selection_dependent")),
                "loss": _as_list(grid.get("loss", "selection_dependent")),
                "resolution": _as_list(grid.get("resolution", 224)),
                "seed": _as_list(grid.get("seed", 2026)),
            }
            for dataset, task, model, loss, resolution, seed in product(*values.values()):
                planned.append(
                    RunSpec(
                        dataset=str(dataset),
                        task=str(task),
                        model=str(model),
                        loss=str(loss),
                        resolution=(
                            int(resolution)
                            if isinstance(resolution, int)
                            or str(resolution).isdigit()
                            else str(resolution)
                        ),
                        seed=int(seed),
                        stage=str(stage["id"]),
                    )
                )
    run_ids = [run.run_id for run in planned]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Program config generates duplicate deterministic run IDs")
    return planned


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()
