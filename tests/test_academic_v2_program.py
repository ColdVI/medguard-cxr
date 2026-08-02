"""Deterministic planning, run state, and resume proof tests."""

from pathlib import Path

from medguard.program.executor import execute_contract_smoke
from medguard.program.planner import load_program, plan_runs
from medguard.program.registry import RunRegistry


def test_program_has_deterministic_unique_bounded_run_ids() -> None:
    program = load_program("configs/programs/academic_v2_full.yaml")

    first = plan_runs(program, "full")
    second = plan_runs(program, "full")

    assert [item.run_id for item in first] == [item.run_id for item in second]
    assert len({item.run_id for item in first}) == len(first)
    assert {item.stage for item in first} >= {"E0", "E1", "E7", "L2", "V1"}


def test_synthetic_contract_proof_resumes_without_fake_metrics(tmp_path: Path) -> None:
    program = load_program("configs/programs/academic_v2_full.yaml")
    spec = plan_runs(program, "synthetic")[0]
    registry = RunRegistry(tmp_path)

    first = execute_contract_smoke(spec, registry, resume=True)
    first_status = registry.read_status(spec.run_id)
    second = execute_contract_smoke(spec, registry, resume=True)

    assert first == second
    assert first["smoke_status"] == "passed"
    assert first["research_status"] == "not_run"
    assert first["performance_metrics_emitted"] is False
    assert first_status["status"] == "skipped_by_design"
    assert len(registry.summary()) == 1
    assert not (registry.run_directory(spec.run_id) / "DONE.json").exists()
