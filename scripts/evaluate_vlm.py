"""Evaluate Phase 4B rule-based, zero-shot VLM, and LoRA VLM VQA paths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from PIL import Image

from medguard.api.schemas import SMOKE_WARNING, EvidencePayload
from medguard.data.nih import NIH_LABELS
from medguard.models.vlm import (
    check_consistency,
    dependency_status,
    load_vlm,
    mentions_unsupported_finding,
    phase4b_system_prompt,
    vlm_response_or_fallback,
)
from medguard.models.vlm import (
    is_available as vlm_is_available,
)
from medguard.safety.abstention import PredictionRecord, load_thresholds_from_config
from medguard.vqa.rule_based import (
    answer_question,
    build_default_provenance,
    thresholds_config_with_classes,
)
from medguard.vqa.templates import template_matches

PHASE = "4B"
Backend = Literal["zero_shot", "lora", "compare"]


def main(argv: list[str] | None = None) -> int:
    """Run VLM comparison evaluation."""

    args = _parse_args(argv)
    config = _load_yaml(args.config)
    test_jsonl = Path(args.test_jsonl or config["vqa"]["output"]["test_jsonl"])
    output = Path(args.output or _default_output(config, args.backend))
    report = evaluate_vlm(
        config=config,
        test_jsonl=test_jsonl,
        backend=args.backend,
        adapter=Path(args.adapter) if args.adapter else None,
        limit=args.limit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.backend == "zero_shot":
        zero_path = Path(
            config["vlm"]["zero_shot"].get("output", "results/vlm_zero_shot_eval.json")
        )
        if zero_path != output:
            zero_path.parent.mkdir(parents=True, exist_ok=True)
            zero_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote VLM evaluation report to {output}")
    return 0


def evaluate_vlm(
    config: dict[str, Any],
    test_jsonl: Path,
    backend: Backend = "compare",
    adapter: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate rule-based and optional VLM paths on one test JSONL."""

    if not test_jsonl.exists():
        return _blocked_report(config, test_jsonl, f"test_jsonl_missing:{test_jsonl}")
    records = _read_jsonl(test_jsonl)
    if limit is not None:
        records = records[:limit]
    thresholds = load_thresholds_from_config(
        thresholds_config_with_classes(_load_yaml("configs/calibration.yaml"))
    )
    provenance = build_default_provenance(is_smoke=True)
    rule_records = [_evaluate_rule_record(record, thresholds, provenance) for record in records]
    comparison: dict[str, Any] = {
        "rule_based": _metrics(rule_records, records, computed=True),
    }
    per_record: list[dict[str, Any]] = [
        {"image_id": item["image_id"], "question": item["question"], "rule_based": item}
        for item in rule_records
    ]

    if backend in {"zero_shot", "compare"}:
        zero_records = _evaluate_vlm_backend(
            config=config,
            records=records,
            thresholds=thresholds,
            provenance=provenance,
            source="vlm_zero_shot",
            adapter=None,
        )
        comparison["vlm_zero_shot"] = _metrics(
            zero_records,
            records,
            computed=all(item.get("available") for item in zero_records),
        )
        _merge_backend(per_record, "vlm_zero_shot", zero_records)
    if backend in {"lora", "compare"}:
        lora_adapter = adapter or Path(
            config["vlm"]["adapter"].get("path", "checkpoints/vlm_lora_adapter")
        )
        lora_records = _evaluate_vlm_backend(
            config=config,
            records=records,
            thresholds=thresholds,
            provenance=provenance,
            source="vlm_lora",
            adapter=lora_adapter,
        )
        comparison["vlm_lora"] = _metrics(
            lora_records,
            records,
            computed=all(item.get("available") for item in lora_records),
        )
        _merge_backend(per_record, "vlm_lora", lora_records)

    return {
        "mode": "vlm_phase4b_evaluation",
        "WARNING_DO_NOT_USE": SMOKE_WARNING,
        "model_quality_evidence": False,
        "test_jsonl": str(test_jsonl),
        "backend": backend,
        "record_count": len(records),
        "comparison": comparison,
        "per_record": per_record,
        "dependency_status": dependency_status(),
    }


def _evaluate_rule_record(
    record: dict[str, Any],
    thresholds: Any,
    provenance: Any,
) -> dict[str, Any]:
    probs = _probabilities_from_record(record, thresholds.classes)
    evidence = _evidence_from_record(record)
    response = answer_question(
        str(record["question"]),
        probs,
        thresholds,
        provenance=provenance,
        evidence=evidence,
        require_evidence_for_positive=True,
    )
    return {
        "image_id": record.get("image_id"),
        "question": record.get("question"),
        "answer": response.answer,
        "expected_answer": record.get("answer"),
        "source": response.source,
        "abstained": response.abstained,
        "reason": response.reason,
        "exact_match": _normalize(response.answer)
        == _normalize(str(record.get("answer", ""))),
        "template_adherence": template_matches(response.answer, record.get("label_class")),
    }


def _evaluate_vlm_backend(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    thresholds: Any,
    provenance: Any,
    source: Literal["vlm_zero_shot", "vlm_lora"],
    adapter: Path | None,
) -> list[dict[str, Any]]:
    if not vlm_is_available():
        return [
            _unavailable_record(record, source, "optional_vlm_dependencies_missing")
            for record in records
        ]
    if source == "vlm_lora" and (adapter is None or not adapter.exists()):
        return [
            _unavailable_record(record, source, "vlm_lora_adapter_missing")
            for record in records
        ]

    try:
        engine = load_vlm(config)
        engine.source_tag = source
    except Exception as exc:
        return [_unavailable_record(record, source, f"vlm_load_failed:{exc}") for record in records]

    outputs: list[dict[str, Any]] = []
    for record in records:
        probs = _probabilities_from_record(record, thresholds.classes)
        evidence = _evidence_from_record(record)
        fallback = answer_question(
            str(record["question"]),
            probs,
            thresholds,
            provenance=provenance,
            evidence=evidence,
            require_evidence_for_positive=True,
        )
        decision = _decision_from_record(record, thresholds.classes)
        image = _image_for_record(record)
        started = time.perf_counter()
        raw_answer = engine.generate(image, str(record["question"]), phase4b_system_prompt())
        response, filter_result = vlm_response_or_fallback(
            raw_output=raw_answer,
            question=str(record["question"]),
            classifier_decision=decision,
            provenance=provenance,
            evidence=evidence,
            fallback=fallback,
            source=source,
            max_answer_tokens=int(config.get("safety", {}).get("max_answer_tokens", 200)),
        )
        outputs.append(
            {
                "image_id": record.get("image_id"),
                "question": record.get("question"),
                "available": True,
                "raw_vlm_answer": raw_answer,
                "filtered_answer": response.answer,
                "filter_result": filter_result.filter_result,
                "filter_reason": filter_result.reason,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "source": response.source,
                "abstained": response.abstained,
                "reason": response.reason,
                "exact_match": _normalize(response.answer)
                == _normalize(str(record.get("answer", ""))),
                "template_adherence": template_matches(response.answer, record.get("label_class")),
                "hallucination": mentions_unsupported_finding(response.answer),
                "classifier_consistent": check_consistency(response.answer, decision),
            }
        )
    return outputs


def _metrics(
    evaluated: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    computed: bool,
) -> dict[str, Any]:
    if not computed or not evaluated:
        return {
            "computed": False,
            "exact_match": None,
            "template_adherence": None,
            "rejection_rate": None,
            "hallucination_rate": None,
            "abstention_correctness": None,
            "classifier_consistency": None,
            "safety_filter_pass_rate": None,
        }
    n = max(len(evaluated), 1)
    rejection_subset = [
        item
        for item, source in zip(evaluated, expected, strict=False)
        if source.get("label_kind") in {"unsupported_concept", "diagnosis_request"}
    ]
    return {
        "computed": True,
        "exact_match": _mean_bool(item.get("exact_match") for item in evaluated),
        "template_adherence": _mean_bool(item.get("template_adherence") for item in evaluated),
        "rejection_rate": _mean_bool(item.get("abstained") for item in rejection_subset)
        if rejection_subset
        else None,
        "hallucination_rate": _mean_bool(item.get("hallucination", False) for item in evaluated),
        "abstention_correctness": _mean_bool(
            bool(item.get("abstained")) == bool(source.get("model_abstained"))
            for item, source in zip(evaluated, expected, strict=False)
        ),
        "classifier_consistency": _mean_bool(
            item.get("classifier_consistent", True) for item in evaluated
        ),
        "safety_filter_pass_rate": sum(
            1 for item in evaluated if item.get("filter_result", "pass") == "pass"
        )
        / n,
    }


def _probabilities_from_record(record: dict[str, Any], classes: list[str]) -> np.ndarray:
    probs = np.full(len(classes), 0.10, dtype=np.float64)
    class_name = record.get("label_class")
    if class_name in classes:
        index = classes.index(str(class_name))
        if record.get("model_abstained"):
            probs[index] = 0.50
        else:
            probs[index] = float(record.get("model_confidence", 0.10))
    return probs


def _decision_from_record(record: dict[str, Any], classes: list[str]) -> PredictionRecord:
    class_name = str(record.get("label_class") or "Pneumonia")
    if class_name not in classes:
        class_name = "Pneumonia"
    return PredictionRecord(
        image_id=str(record.get("image_id", "")),
        class_name=class_name,
        prediction=record.get("model_prediction"),
        confidence=float(record.get("model_confidence", 0.0)),
        abstained=bool(record.get("model_abstained", True)),
        reason=str(record.get("abstention_reason", "")),
    )


def _evidence_from_record(record: dict[str, Any]) -> EvidencePayload | None:
    if not record.get("evidence_available"):
        return None
    class_name = record.get("evidence_class") or record.get("label_class")
    if class_name not in NIH_LABELS:
        return None
    return EvidencePayload(
        class_name=str(class_name),
        cam_uri=None,
        bbox_normalized=None,
        cam_method="gradcam",
    )


def _image_for_record(record: dict[str, Any]) -> Image.Image:
    path = Path(str(record.get("image_path") or record.get("path") or ""))
    if path.exists():
        return Image.open(path).convert("RGB")
    return Image.new("RGB", (224, 224), color=(32, 32, 32))


def _unavailable_record(record: dict[str, Any], source: str, reason: str) -> dict[str, Any]:
    return {
        "image_id": record.get("image_id"),
        "question": record.get("question"),
        "available": False,
        "source": source,
        "raw_vlm_answer": None,
        "filtered_answer": None,
        "filter_result": "unavailable",
        "filter_reason": reason,
    }


def _merge_backend(
    per_record: list[dict[str, Any]],
    key: str,
    backend_records: list[dict[str, Any]],
) -> None:
    for item, backend_record in zip(per_record, backend_records, strict=False):
        item[key] = backend_record


def _blocked_report(config: dict[str, Any], test_jsonl: Path, reason: str) -> dict[str, Any]:
    return {
        "mode": "vlm_phase4b_evaluation_blocked",
        "WARNING_DO_NOT_USE": SMOKE_WARNING,
        "model_quality_evidence": False,
        "reason": reason,
        "test_jsonl": str(test_jsonl),
        "comparison": {
            "rule_based": {"computed": False},
            "vlm_zero_shot": {"computed": False},
            "vlm_lora": {"computed": False},
        },
        "per_record": [],
        "dependency_status": dependency_status(),
        "base_model": config.get("vlm", {}).get("base_model"),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _mean_bool(values: Any) -> float:
    items = [bool(value) for value in values]
    return float(sum(items) / len(items)) if items else 0.0


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _default_output(config: dict[str, Any], backend: Backend) -> str:
    if backend == "zero_shot":
        return str(config["vlm"]["zero_shot"].get("output", "results/vlm_zero_shot_eval.json"))
    return "results/vlm_comparison.json"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vlm_lora.yaml")
    parser.add_argument("--test-jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/baseline_nih_best.pt")
    parser.add_argument("--adapter", default="checkpoints/vlm_lora_adapter")
    parser.add_argument("--output")
    parser.add_argument("--backend", choices=["zero_shot", "lora", "compare"], default="compare")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
