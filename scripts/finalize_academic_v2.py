#!/usr/bin/env python3
"""Strict Academic V2 finalization gate; never promotes incomplete research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medguard.data.cache import atomic_write_text
from medguard.program.registry import RunRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    registry = RunRegistry(workspace)
    statuses = registry.summary()
    if args.smoke:
        passed = bool(statuses) and all(
            item.get("artifact_kind") == "smoke" and item.get("smoke_status") == "passed"
            for item in statuses
        )
        report = {
            "finalization": "smoke_passed" if passed else "smoke_failed",
            "research_complete": False,
            "release_created": False,
            "warning": "Synthetic finalization validates plumbing only.",
        }
        code = 0 if passed else 1
    else:
        completed = [item for item in statuses if item["status"] == "completed"]
        incomplete = [item for item in statuses if item["status"] != "completed"]
        valid_done = all(
            (registry.run_directory(item["run_id"]) / "DONE.json").is_file()
            for item in completed
        )
        research_complete = bool(statuses) and not incomplete and valid_done
        report = {
            "finalization": "completed" if research_complete else "incomplete",
            "research_complete": research_complete,
            "release_created": False,
            "completed_runs": len(completed),
            "incomplete_runs": len(incomplete),
            "blocked_runs": sum(
                item["status"] == "blocked_external_access" for item in statuses
            ),
            "missing_or_invalid_done_markers": not valid_done,
        }
        code = 0 if research_complete or not args.strict else 2
    target = workspace / "results_v2" / "finalization_status.json"
    atomic_write_text(target, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
