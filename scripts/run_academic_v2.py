#!/usr/bin/env python3
"""Run or resume the MEDGUARD-CXR Academic V2 research program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medguard.data.cache import atomic_write_text
from medguard.program.executor import execute_contract_smoke, register_research_plan
from medguard.program.planner import load_program, plan_runs
from medguard.program.registry import RunRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/programs/academic_v2_full.yaml")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--profile", choices=("full", "synthetic"), default="full")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    program = load_program(args.config)
    specs = plan_runs(program, args.profile)
    registry = RunRegistry(workspace)
    if args.profile == "synthetic":
        proof = execute_contract_smoke(specs[0], registry, args.resume)
        result = {"profile": "synthetic", "proof": proof, "runs": registry.summary()}
    else:
        reports = register_research_plan(specs, registry, workspace)
        result = {
            "profile": "full",
            "planned_runs": len(specs),
            "status_counts": _status_counts(reports),
            "warning": (
                "First-slice executor registers/preflights runs; "
                "training stages remain pending."
            ),
        }
    summary_path = workspace / "results_v2" / "orchestrator_summary.json"
    atomic_write_text(summary_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _status_counts(reports: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report in reports:
        status = str(report["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
