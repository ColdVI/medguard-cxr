"""Optional Phase 4B QLoRA training entrypoint.

The script is intentionally config-driven and writes an explicit blocker report
when GPU resources, local VQA JSONL files, or optional VLM dependencies are not
available. It never merges LoRA adapters into base model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from medguard.api.schemas import SMOKE_WARNING
from medguard.models.vlm import dependency_status, is_available

PHASE = "4B"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse VLM training arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vlm_lora.yaml")
    parser.add_argument("--train-jsonl")
    parser.add_argument("--val-jsonl")
    parser.add_argument("--smoke", action="store_true", help="Validate setup without training.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run optional QLoRA training or emit a blocker report."""

    args = parse_args(argv)
    config = _load_yaml(args.config)
    vlm_cfg = config.get("vlm", {})
    training_cfg = vlm_cfg.get("training", {})
    output_path = Path(training_cfg.get("output_report", "results/vlm_lora_train.json"))
    train_path = Path(args.train_jsonl or config["vqa"]["output"]["train_jsonl"])
    val_path = Path(args.val_jsonl or config["vqa"]["output"]["val_jsonl"])
    adapter_path = Path(vlm_cfg.get("adapter", {}).get("path", "checkpoints/vlm_lora_adapter"))

    blocker = _training_blocker(config, train_path, val_path, smoke=args.smoke)
    if blocker is not None:
        report = _blocked_report(
            config=config,
            adapter_path=adapter_path,
            train_path=train_path,
            val_path=val_path,
            blocker=blocker,
        )
        _write_report(output_path, report)
        print(f"QLoRA training blocked: {blocker}")
        print(f"Wrote training report to {output_path}")
        return 0

    report = _run_training(config, train_path, val_path, adapter_path)
    _write_report(output_path, report)
    print(f"Wrote training report to {output_path}")
    return 0


def _training_blocker(
    config: dict[str, Any],
    train_path: Path,
    val_path: Path,
    smoke: bool,
) -> str | None:
    if smoke:
        return "smoke_check_requested_no_training"
    if not config.get("vlm", {}).get("training", {}).get("enabled", False):
        return "training_disabled_in_config"
    if not train_path.exists() or not val_path.exists():
        return "vqa_train_or_val_jsonl_missing"
    if not is_available():
        missing = ",".join(dependency_status()["missing"])
        return f"optional_vlm_dependencies_missing:{missing}"
    try:
        import torch
    except ImportError:
        return "torch_missing"
    if not torch.cuda.is_available():
        return "cuda_gpu_unavailable"
    return None


def _run_training(
    config: dict[str, Any],
    train_path: Path,
    val_path: Path,
    adapter_path: Path,
) -> dict[str, Any]:
    """Run a minimal adapter-only QLoRA training loop when resources exist.

    The heavyweight imports and CUDA checks happen only after blocker checks.
    """

    from medguard.models.vlm import load_vlm

    train_records = _read_jsonl(train_path)
    val_records = _read_jsonl(val_path)
    engine = load_vlm(config)
    adapter_path.mkdir(parents=True, exist_ok=True)
    adapter_config = adapter_path / "adapter_config.json"
    adapter_config.write_text(
        json.dumps(
            {
                "base_model": config["vlm"]["base_model"],
                "lora": config["vlm"]["lora"],
                "note": "adapter-only checkpoint directory initialized by Phase 4B",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if hasattr(engine.model, "save_pretrained"):
        engine.model.save_pretrained(adapter_path)

    return {
        "mode": "vlm_lora_training",
        "WARNING_DO_NOT_USE": SMOKE_WARNING,
        "model_quality_evidence": False,
        "base_model": config["vlm"]["base_model"],
        "adapter_path": str(adapter_path),
        "adapter_sha256": _directory_sha256(adapter_path),
        "epochs_completed": 0,
        "best_val_loss": None,
        "early_stopped": True,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "seed": int(config.get("seed", 2026)),
        "status": "adapter_initialized_training_loop_blocked_pending_full_gpu_run",
    }


def _blocked_report(
    config: dict[str, Any],
    adapter_path: Path,
    train_path: Path,
    val_path: Path,
    blocker: str,
) -> dict[str, Any]:
    return {
        "mode": "vlm_lora_training",
        "WARNING_DO_NOT_USE": SMOKE_WARNING,
        "model_quality_evidence": False,
        "base_model": config.get("vlm", {}).get("base_model", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "adapter_path": str(adapter_path),
        "epochs_completed": 0,
        "best_val_loss": None,
        "early_stopped": True,
        "train_records": _count_jsonl(train_path),
        "val_records": _count_jsonl(val_path),
        "seed": int(config.get("seed", 2026)),
        "status": "blocked",
        "blocked_reason": blocker,
        "dependency_status": dependency_status(),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


if __name__ == "__main__":
    raise SystemExit(main())
