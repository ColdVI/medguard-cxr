"""Atomic filesystem run registry with checksum-aware resume semantics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medguard.data.cache import atomic_write_text
from medguard.program.planner import RunSpec
from medguard.program.state import RunStatus, assert_transition


class RunRegistry:
    """One directory per run; v1 result paths are never used or overwritten."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.runs_root = self.workspace / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def register(self, spec: RunSpec) -> dict[str, Any]:
        directory = self.run_directory(spec.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "config.json"
        expected = json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n"
        if config_path.exists() and config_path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"Run ID collision with different config: {spec.run_id}")
        if not config_path.exists():
            atomic_write_text(config_path, expected)
        status_path = directory / "status.json"
        if not status_path.exists():
            self._write_status(spec.run_id, RunStatus.PENDING, detail="registered")
        return self.read_status(spec.run_id)

    def transition(
        self,
        run_id: str,
        target: RunStatus,
        *,
        detail: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = RunStatus(self.read_status(run_id)["status"])
        assert_transition(current, target)
        return self._write_status(run_id, target, detail=detail, extra=extra)

    def read_status(self, run_id: str) -> dict[str, Any]:
        path = self.run_directory(run_id) / "status.json"
        if not path.exists():
            raise KeyError(f"Run is not registered: {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def run_directory(self, run_id: str) -> Path:
        return self.runs_root / run_id

    def resumable(self, run_id: str) -> bool:
        """Completed research runs require validated DONE; smoke proof uses its own marker."""

        status = RunStatus(self.read_status(run_id)["status"])
        if status is RunStatus.COMPLETED:
            return not (self.run_directory(run_id) / "DONE.json").exists()
        return status in {
            RunStatus.PENDING,
            RunStatus.RUNNING,
            RunStatus.FAILED,
            RunStatus.BLOCKED_EXTERNAL_ACCESS,
        }

    def summary(self) -> list[dict[str, Any]]:
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.runs_root.glob("*/status.json"))
        ]

    def _write_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        detail: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": run_id,
            "status": status.value,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(extra or {})
        atomic_write_text(
            self.run_directory(run_id) / "status.json",
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        return payload
