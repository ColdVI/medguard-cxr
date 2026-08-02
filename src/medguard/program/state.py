"""Strict run states and legal transition rules."""

from __future__ import annotations

from enum import Enum


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED_EXTERNAL_ACCESS = "blocked_external_access"
    SKIPPED_BY_SELECTION = "skipped_by_selection"
    SKIPPED_BY_DESIGN = "skipped_by_design"


TERMINAL_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.BLOCKED_EXTERNAL_ACCESS,
    RunStatus.SKIPPED_BY_SELECTION,
    RunStatus.SKIPPED_BY_DESIGN,
}

ALLOWED_TRANSITIONS = {
    RunStatus.PENDING: {
        RunStatus.RUNNING,
        RunStatus.BLOCKED_EXTERNAL_ACCESS,
        RunStatus.SKIPPED_BY_SELECTION,
        RunStatus.SKIPPED_BY_DESIGN,
        RunStatus.FAILED,
    },
    RunStatus.RUNNING: {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.BLOCKED_EXTERNAL_ACCESS,
        RunStatus.SKIPPED_BY_DESIGN,
    },
    RunStatus.FAILED: {RunStatus.RUNNING},
    RunStatus.BLOCKED_EXTERNAL_ACCESS: {RunStatus.RUNNING},
    RunStatus.SKIPPED_BY_SELECTION: set(),
    RunStatus.SKIPPED_BY_DESIGN: set(),
    RunStatus.COMPLETED: set(),
}


def assert_transition(current: RunStatus, target: RunStatus) -> None:
    if current == target:
        return
    if target not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"Illegal run status transition: {current.value} -> {target.value}")
