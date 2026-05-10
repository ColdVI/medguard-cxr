"""Optional Phase 4B QLoRA training entrypoint.

The script is intentionally config-driven and writes an explicit blocker report
when GPU resources, local VQA JSONL files, or optional VLM dependencies are not
available. When explicitly enabled, it runs an adapter-only PEFT/QLoRA training
loop and never merges LoRA adapters into base model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml

from medguard.api.schemas import SAFETY_DISCLAIMER, SMOKE_WARNING
from medguard.data.nih import NIH_LABELS
from medguard.data.rsna import read_rsna_image
from medguard.models.vlm import dependency_status, is_available, phase4b_system_prompt

PHASE = "4B"
VLM_SOURCE = "vlm_lora"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse VLM training arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vlm_lora.yaml")
    parser.add_argument("--train-jsonl")
    parser.add_argument("--val-jsonl")
    parser.add_argument("--smoke", action="store_true", help="Validate setup without training.")
    parser.add_argument(
        "--enable-training",
        action="store_true",
        help="Override config and run training when all blockers are clear.",
    )
    parser.add_argument("--max-steps", type=int, help="Override training.max_steps.")
    parser.add_argument("--limit-train", type=int, help="Limit train records for a short run.")
    parser.add_argument("--limit-val", type=int, help="Limit validation records for a short run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run optional QLoRA training or emit a blocker report."""

    args = parse_args(argv)
    config = _load_yaml(args.config)
    _apply_cli_overrides(config, args)
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
    """Run an adapter-only QLoRA training loop when resources exist.

    The heavyweight imports and CUDA checks happen only after blocker checks.
    """

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Trainer

    train_records = _read_jsonl(train_path)
    val_records = _read_jsonl(val_path)
    train_records = _limit_records(
        train_records,
        config.get("vlm", {}).get("training", {}).get("max_train_samples"),
    )
    val_records = _limit_records(
        val_records,
        config.get("vlm", {}).get("training", {}).get("max_val_samples"),
    )
    if not train_records:
        raise ValueError("QLoRA training requires at least one train JSONL record.")
    if not val_records:
        raise ValueError("QLoRA training requires at least one validation JSONL record.")

    vlm_cfg = config.get("vlm", {})
    training_cfg = vlm_cfg.get("training", {})
    lora_cfg = vlm_cfg.get("lora", {})
    model_name = str(vlm_cfg.get("base_model", "Qwen/Qwen2.5-VL-3B-Instruct"))
    compute_dtype = _torch_dtype(torch, str(vlm_cfg.get("compute_dtype", "float16")))
    quantization_config = None
    if bool(vlm_cfg.get("load_in_4bit", True)):
        quant_cfg = dict(vlm_cfg.get("quantization", {}))
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(
                quant_cfg.get("bnb_4bit_use_double_quant", True)
            ),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    trust_remote_code = bool(vlm_cfg.get("trust_remote_code", True))
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_class = _vlm_model_class()
    model = model_class.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
    )
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "config"):
        model.config.use_cache = False

    started = time.perf_counter()
    trainer = Trainer(
        model=model,
        args=_training_arguments(config, adapter_path),
        train_dataset=_JSONLRecordDataset(train_records),
        eval_dataset=_JSONLRecordDataset(val_records),
        data_collator=_VLMDataCollator(
            processor=processor,
            max_seq_length=int(training_cfg.get("max_seq_length", 512)),
        ),
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path / "processor")

    report = {
        "mode": "vlm_lora_training",
        "model_quality_evidence": False,
        "base_model": model_name,
        "adapter_path": str(adapter_path),
        "adapter_sha256": _directory_sha256(adapter_path),
        "epochs_completed": float(getattr(trainer.state, "epoch", 0.0) or 0.0),
        "best_val_loss": _metric_float(eval_metrics.get("eval_loss")),
        "early_stopped": False,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "seed": int(config.get("seed", 2026)),
        "status": "completed",
        "train_metrics": _jsonable_metrics(train_result.metrics),
        "eval_metrics": _jsonable_metrics(eval_metrics),
        "training_runtime_seconds": round(time.perf_counter() - started, 3),
        "notes": [
            "Adapter-only PEFT/QLoRA training completed.",
            "This report is training evidence, not clinical validation.",
            "Run evaluate_vlm.py before claiming VLM answer quality.",
        ],
    }
    if _classifier_checkpoint_is_smoke(config):
        report["WARNING_DO_NOT_USE"] = SMOKE_WARNING
    return report


class _JSONLRecordDataset:
    """Small Trainer-compatible dataset over parsed VQA JSONL records."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class _VLMDataCollator:
    """Build masked language-model batches for image/question/JSON-answer records."""

    def __init__(self, processor: Any, max_seq_length: int) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length
        tokenizer = getattr(processor, "tokenizer", processor)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        images = [_load_training_image(record) for record in records]
        prompts = [_prompt_text(record) for record in records]
        targets = [_target_text(record) for record in records]
        texts = [prompt + target for prompt, target in zip(prompts, targets, strict=True)]
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )
        labels = batch["input_ids"].clone()
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        for row_index, (prompt, image) in enumerate(zip(prompts, images, strict=True)):
            prompt_inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
            )
            prompt_len = int(prompt_inputs["input_ids"].shape[-1])
            labels[row_index, : min(prompt_len, labels.shape[-1])] = -100
        batch["labels"] = labels
        return batch


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


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    vlm_cfg = config.setdefault("vlm", {})
    training_cfg = vlm_cfg.setdefault("training", {})
    if args.enable_training:
        training_cfg["enabled"] = True
    if args.max_steps is not None:
        training_cfg["max_steps"] = args.max_steps
    if args.limit_train is not None:
        training_cfg["max_train_samples"] = args.limit_train
    if args.limit_val is not None:
        training_cfg["max_val_samples"] = args.limit_val


def _training_arguments(config: dict[str, Any], adapter_path: Path) -> Any:
    from transformers import TrainingArguments

    training_cfg = config.get("vlm", {}).get("training", {})
    max_steps = training_cfg.get("max_steps")
    args: dict[str, Any] = {
        "output_dir": str(adapter_path),
        "overwrite_output_dir": bool(training_cfg.get("overwrite_output_dir", True)),
        "num_train_epochs": float(training_cfg.get("epochs", 3)),
        "per_device_train_batch_size": int(training_cfg.get("batch_size", 4)),
        "per_device_eval_batch_size": int(training_cfg.get("eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("grad_accum", 4)),
        "learning_rate": float(training_cfg.get("learning_rate", 2.0e-4)),
        "warmup_steps": int(training_cfg.get("warmup_steps", 100)),
        "lr_scheduler_type": str(training_cfg.get("scheduler", "cosine")),
        "logging_steps": int(training_cfg.get("logging_steps", 10)),
        "save_strategy": str(training_cfg.get("save_strategy", "epoch")),
        "eval_strategy": str(training_cfg.get("eval_strategy", "epoch")),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "remove_unused_columns": False,
        "report_to": [],
        "fp16": str(config.get("vlm", {}).get("compute_dtype", "float16")) == "float16",
        "bf16": str(config.get("vlm", {}).get("compute_dtype", "")) == "bfloat16",
        "optim": str(training_cfg.get("optim", "paged_adamw_8bit")),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", True)),
    }
    if max_steps not in {None, "", 0}:
        args["max_steps"] = int(max_steps)
        args["save_strategy"] = str(training_cfg.get("save_strategy", "steps"))
        args["eval_strategy"] = str(training_cfg.get("eval_strategy", "steps"))
        args["eval_steps"] = int(training_cfg.get("eval_steps", max(1, int(max_steps) // 2)))
        args["save_steps"] = int(training_cfg.get("save_steps", max(1, int(max_steps) // 2)))
    return TrainingArguments(**args)


def _vlm_model_class() -> Any:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:  # pragma: no cover - depends on installed transformers.
        from transformers import AutoModelForVision2Seq as ModelClass
    return ModelClass


def _torch_dtype(torch: Any, name: str) -> Any:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported VLM compute dtype: {name}")


def _limit_records(
    records: list[dict[str, Any]],
    limit: Any,
) -> list[dict[str, Any]]:
    if limit in {None, "", 0}:
        return records
    return records[: int(limit)]


def _prompt_text(record: dict[str, Any]) -> str:
    return (
        "<|im_start|>system\n"
        f"{phase4b_system_prompt(NIH_LABELS)}<|im_end|>\n"
        "<|im_start|>user\n"
        "<image>\n"
        f"{record['question']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _target_text(record: dict[str, Any]) -> str:
    return json.dumps(_target_payload(record), ensure_ascii=False, sort_keys=True) + "<|im_end|>"


def _target_payload(record: dict[str, Any]) -> dict[str, Any]:
    label_kind = str(record.get("label_kind", ""))
    abstained = bool(record.get("model_abstained")) or label_kind in {
        "unsupported_concept",
        "diagnosis_request",
    }
    reason = _reason_for_target(record, abstained=abstained)
    return {
        "question": str(record["question"]),
        "answer": str(record["answer"]),
        "confidence": float(record.get("model_confidence", 0.0)),
        "evidence": _target_evidence(record),
        "abstained": abstained,
        "reason": reason,
        "safety_disclaimer": SAFETY_DISCLAIMER,
        "source": VLM_SOURCE,
    }


def _reason_for_target(record: dict[str, Any], abstained: bool) -> str:
    label_kind = str(record.get("label_kind", ""))
    if label_kind in {"unsupported_concept", "diagnosis_request"}:
        return label_kind
    if abstained:
        reason = str(record.get("abstention_reason") or "low_confidence_band")
        return reason if reason else "low_confidence_band"
    return ""


def _target_evidence(record: dict[str, Any]) -> dict[str, Any] | None:
    if not record.get("evidence_available"):
        return None
    class_name = record.get("evidence_class") or record.get("label_class")
    if class_name not in NIH_LABELS:
        return None
    return {
        "class_name": class_name,
        "cam_uri": record.get("cam_uri"),
        "bbox_normalized": record.get("bbox_normalized"),
        "cam_method": "gradcam",
    }


def _load_training_image(record: dict[str, Any]) -> Any:
    path = Path(str(record.get("image_path") or record.get("path") or ""))
    if not path.exists():
        raise FileNotFoundError(f"VQA record image not found: {path}")
    if path.suffix.lower() in {".dcm", ".dicom"}:
        return read_rsna_image(path).convert("RGB")
    from PIL import Image

    return Image.open(path).convert("RGB")


def _metric_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _jsonable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, int | float | str | bool) or value is None:
            output[key] = value
        else:
            try:
                output[key] = float(value)
            except (TypeError, ValueError):
                output[key] = str(value)
    return output


def _classifier_checkpoint_is_smoke(config: dict[str, Any]) -> bool:
    return bool(
        config.get("vlm", {}).get("training", {}).get("classifier_checkpoint_is_smoke", True)
    )


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
