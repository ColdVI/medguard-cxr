"""Generate synthetic CXR-QA records from classifier-grounded selective outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from medguard.api.schemas import SMOKE_WARNING
from medguard.data.nih import NIH_LABELS
from medguard.safety.abstention import (
    AbstentionThresholds,
    apply_abstention,
    load_thresholds_from_config,
)
from medguard.vqa.rule_based import thresholds_config_with_classes
from medguard.vqa.templates import (
    DIAGNOSIS_REQUEST_QUESTIONS,
    UNSUPPORTED_CONCEPT_QUESTIONS,
    answer_for_label_kind,
    confidence_answer,
    localization_answer,
    multi_finding_answer,
    validate_qa_record,
)

PHASE = "4"


def main() -> None:
    """Run the JSONL generation CLI."""

    args = _parse_args()
    config = _load_yaml(args.config)
    baseline = _load_yaml(args.baseline_config)
    records = read_manifest(args.input_manifest)
    if args.limit is not None:
        records = records[: args.limit]
    thresholds = load_thresholds_from_config(
        thresholds_config_with_classes(_load_yaml("configs/calibration.yaml"))
    )
    output_base = Path(args.output)
    generated = generate_dataset_records(
        manifest_records=records,
        thresholds=thresholds,
        seed=args.seed,
        n_distractors=args.n_distractors,
        include_uncertain=args.include_uncertain,
        calibrator_path=args.calibrator,
        class_names=list(baseline.get("data", {}).get("labels", NIH_LABELS)),
        is_smoke=not (Path(args.checkpoint).exists() and Path(args.calibrator).exists()),
    )
    splits = patient_disjoint_split(generated, seed=args.seed)
    write_split_jsonl(splits, output_base)
    _write_generation_summary(config, output_base, splits)


def generate_dataset_records(
    manifest_records: Sequence[dict[str, str]],
    thresholds: AbstentionThresholds,
    seed: int = 2026,
    n_distractors: int = 2,
    include_uncertain: bool = True,
    calibrator_path: str | Path = "calibrators/nih_temp_scaling.pkl",
    class_names: list[str] | None = None,
    is_smoke: bool = True,
) -> list[dict[str, Any]]:
    """Generate QA records for a manifest without using test data for fitting."""

    rng = random.Random(seed)
    classes = class_names or NIH_LABELS
    output: list[dict[str, Any]] = []
    calibrator_version = _artifact_version(calibrator_path)
    for row_index, row in enumerate(manifest_records):
        image_id = row.get("image_id") or Path(row["path"]).name
        image_path = row["path"]
        patient_id = row.get("patient_id") or infer_patient_id(image_id)
        provenance = row.get("image_provenance") or infer_image_provenance(image_path, is_smoke)
        probs = _smoke_probabilities(row_index, len(classes)) if is_smoke else _smoke_probabilities(
            row_index,
            len(classes),
        )
        decisions = apply_abstention(probs.reshape(1, -1), thresholds)[0]
        negatives: list[dict[str, Any]] = []

        for class_index, decision in enumerate(decisions):
            class_name = classes[class_index]
            if decision.abstained:
                if include_uncertain:
                    output.append(
                        _qa_record(
                            row,
                            image_id,
                            image_path,
                            patient_id,
                            class_name,
                            "uncertain",
                            None,
                            decision.prediction,
                            0.0,
                            True,
                            decision.reason,
                            calibrator_version,
                            provenance,
                            is_smoke,
                        )
                    )
                continue
            if decision.prediction == 1:
                output.append(
                    _qa_record(
                        row,
                        image_id,
                        image_path,
                        patient_id,
                        class_name,
                        "positive",
                        1,
                        1,
                        decision.confidence,
                        False,
                        "",
                        calibrator_version,
                        provenance,
                        is_smoke,
                    )
                )
            else:
                negatives.append(
                    _qa_record(
                        row,
                        image_id,
                        image_path,
                        patient_id,
                        class_name,
                        "negative",
                        0,
                        0,
                        decision.confidence,
                        False,
                        "",
                        calibrator_version,
                        provenance,
                        is_smoke,
                    )
                )
        rng.shuffle(negatives)
        output.extend(negatives[:n_distractors])
        output.append(
            _scope_record(
                row,
                image_id,
                image_path,
                patient_id,
                rng.choice(UNSUPPORTED_CONCEPT_QUESTIONS),
                "unsupported_concept",
                calibrator_version,
                provenance,
                is_smoke,
            )
        )
        output.append(
            _confidence_record(
                row=row,
                image_id=image_id,
                image_path=image_path,
                patient_id=patient_id,
                class_name=classes[0],
                confidence=float(decisions[0].confidence),
                calibrator_version=calibrator_version,
                image_provenance=provenance,
                is_smoke=is_smoke,
            )
        )
        positives = [
            decision.class_name
            for decision in decisions
            if not decision.abstained and decision.prediction == 1
        ]
        output.append(
            _multi_finding_record(
                row=row,
                image_id=image_id,
                image_path=image_path,
                patient_id=patient_id,
                positive_classes=positives,
                calibrator_version=calibrator_version,
                image_provenance=provenance,
                is_smoke=is_smoke,
            )
        )
        if _has_rsna_localization(row, provenance):
            output.append(
                _localization_record(
                    row=row,
                    image_id=image_id,
                    image_path=image_path,
                    patient_id=patient_id,
                    label_kind="positive" if "Pneumonia" in positives else "uncertain",
                    calibrator_version=calibrator_version,
                    image_provenance=provenance,
                    is_smoke=is_smoke,
                )
            )
        output.append(
            _scope_record(
                row,
                image_id,
                image_path,
                patient_id,
                rng.choice(DIAGNOSIS_REQUEST_QUESTIONS),
                "diagnosis_request",
                calibrator_version,
                provenance,
                is_smoke,
            )
        )

    for record in output:
        validate_qa_record(record)
    return output


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a manifest containing at minimum ``image_id`` and ``path`` columns."""

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"image_id", "path"}.issubset(reader.fieldnames or set()):
            raise ValueError("VQA manifest must contain image_id and path columns.")
        return [dict(row) for row in reader]


def patient_disjoint_split(
    records: Sequence[dict[str, Any]],
    seed: int = 2026,
) -> dict[str, list[dict[str, Any]]]:
    """Split generated QA records by patient_id into 80/10/10 train/val/test."""

    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_patient[str(record["patient_id"])].append(dict(record))
    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)
    n = len(patients)
    train_cut = max(1, int(round(n * 0.8))) if n else 0
    val_cut = min(n, train_cut + max(1, int(round(n * 0.1)))) if n > 2 else train_cut
    split_patients = {
        "train": patients[:train_cut],
        "val": patients[train_cut:val_cut],
        "test": patients[val_cut:],
    }
    return {
        split: [record for patient in split_patients[split] for record in by_patient[patient]]
        for split in ["train", "val", "test"]
    }


def write_split_jsonl(splits: dict[str, list[dict[str, Any]]], output_base: Path) -> None:
    """Write train/val/test JSONL files using the requested output stem."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    base = output_base.with_suffix("")
    for split, records in splits.items():
        path = base.with_name(f"{base.name}.{split}.jsonl")
        with path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


def validate_jsonl_provenance(paths: Sequence[str | Path]) -> None:
    """Raise if JSONL records mix incompatible image provenance values."""

    provenance: set[str] = set()
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if line.strip():
                    provenance.add(str(json.loads(line)["image_provenance"]))
    if len(provenance) > 1:
        raise ValueError("VQA training JSONL must not mix image provenance values.")


def infer_patient_id(image_id: str) -> str:
    """Infer a patient identifier from common CXR image IDs."""

    return image_id.split("_", maxsplit=1)[0].split(".", maxsplit=1)[0]


def infer_image_provenance(path: str, is_smoke: bool) -> str:
    """Infer image provenance from a manifest path."""

    lowered = path.lower()
    if is_smoke:
        return "synthetic_smoke"
    if "rsna" in lowered:
        return "rsna"
    if "nih" in lowered:
        return "nih"
    return "synthetic_smoke"


def _qa_record(
    row: dict[str, str],
    image_id: str,
    image_path: str,
    patient_id: str,
    class_name: str,
    label_kind: str,
    ground_truth_label: int | None,
    model_prediction: int | None,
    model_confidence: float,
    model_abstained: bool,
    abstention_reason: str,
    calibrator_version: str,
    image_provenance: str,
    is_smoke: bool,
) -> dict[str, Any]:
    record = {
        "image_id": image_id,
        "image_path": image_path,
        "patient_id": patient_id,
        "question": f"Is there evidence of {class_name.replace('_', ' ')}?",
        "answer": answer_for_label_kind(label_kind, class_name),
        "label_class": class_name,
        "label_kind": label_kind,
        "ground_truth_label": ground_truth_label,
        "model_prediction": model_prediction,
        "model_confidence": round(float(model_confidence), 6),
        "model_abstained": model_abstained,
        "abstention_reason": abstention_reason,
        "calibrator_version": calibrator_version,
        "image_provenance": image_provenance,
    }
    record.update(_phase4b_metadata(row, class_name, image_provenance))
    record.update({key: value for key, value in row.items() if key not in record})
    if is_smoke:
        record["WARNING_DO_NOT_USE"] = SMOKE_WARNING
    return record


def _scope_record(
    row: dict[str, str],
    image_id: str,
    image_path: str,
    patient_id: str,
    question: str,
    label_kind: str,
    calibrator_version: str,
    image_provenance: str,
    is_smoke: bool,
) -> dict[str, Any]:
    record = {
        "image_id": image_id,
        "image_path": image_path,
        "patient_id": patient_id,
        "question": question,
        "answer": answer_for_label_kind(label_kind, n_findings=len(NIH_LABELS)),
        "label_class": None,
        "label_kind": label_kind,
        "ground_truth_label": None,
        "model_prediction": None,
        "model_confidence": 0.0,
        "model_abstained": True,
        "abstention_reason": label_kind,
        "calibrator_version": calibrator_version,
        "image_provenance": image_provenance,
    }
    record.update(_phase4b_metadata(row, None, image_provenance))
    record.update({key: value for key, value in row.items() if key not in record})
    if is_smoke:
        record["WARNING_DO_NOT_USE"] = SMOKE_WARNING
    return record


def _confidence_record(
    row: dict[str, str],
    image_id: str,
    image_path: str,
    patient_id: str,
    class_name: str,
    confidence: float,
    calibrator_version: str,
    image_provenance: str,
    is_smoke: bool,
) -> dict[str, Any]:
    record = {
        "image_id": image_id,
        "image_path": image_path,
        "patient_id": patient_id,
        "question": f"How confident is the model about {class_name.replace('_', ' ')}?",
        "answer": confidence_answer(class_name, confidence),
        "label_class": class_name,
        "label_kind": "confidence_query",
        "ground_truth_label": None,
        "model_prediction": None,
        "model_confidence": round(float(confidence), 6),
        "model_abstained": False,
        "abstention_reason": "",
        "calibrator_version": calibrator_version,
        "image_provenance": image_provenance,
    }
    record.update(_phase4b_metadata(row, class_name, image_provenance))
    record.update({key: value for key, value in row.items() if key not in record})
    if is_smoke:
        record["WARNING_DO_NOT_USE"] = SMOKE_WARNING
    return record


def _multi_finding_record(
    row: dict[str, str],
    image_id: str,
    image_path: str,
    patient_id: str,
    positive_classes: list[str],
    calibrator_version: str,
    image_provenance: str,
    is_smoke: bool,
) -> dict[str, Any]:
    record = {
        "image_id": image_id,
        "image_path": image_path,
        "patient_id": patient_id,
        "question": "What findings are present?",
        "answer": multi_finding_answer(positive_classes),
        "label_class": None,
        "label_kind": "multi_finding",
        "ground_truth_label": None,
        "model_prediction": None,
        "model_confidence": 0.0,
        "model_abstained": not bool(positive_classes),
        "abstention_reason": "" if positive_classes else "low_confidence_band",
        "calibrator_version": calibrator_version,
        "image_provenance": image_provenance,
    }
    record.update(_phase4b_metadata(row, None, image_provenance))
    record.update({key: value for key, value in row.items() if key not in record})
    if is_smoke:
        record["WARNING_DO_NOT_USE"] = SMOKE_WARNING
    return record


def _localization_record(
    row: dict[str, str],
    image_id: str,
    image_path: str,
    patient_id: str,
    label_kind: str,
    calibrator_version: str,
    image_provenance: str,
    is_smoke: bool,
) -> dict[str, Any]:
    answer = (
        localization_answer("Pneumonia", _quadrant_from_row(row))
        if label_kind == "positive"
        else answer_for_label_kind("uncertain", "Pneumonia")
    )
    record = {
        "image_id": image_id,
        "image_path": image_path,
        "patient_id": patient_id,
        "question": "Is there evidence of Pneumonia? Where?",
        "answer": answer,
        "label_class": "Pneumonia",
        "label_kind": f"localization_{label_kind}",
        "ground_truth_label": 1 if label_kind == "positive" else None,
        "model_prediction": 1 if label_kind == "positive" else None,
        "model_confidence": 0.0,
        "model_abstained": label_kind != "positive",
        "abstention_reason": "" if label_kind == "positive" else "low_confidence_band",
        "calibrator_version": calibrator_version,
        "image_provenance": image_provenance,
    }
    record.update(_phase4b_metadata(row, "Pneumonia", image_provenance))
    record.update({key: value for key, value in row.items() if key not in record})
    if is_smoke:
        record["WARNING_DO_NOT_USE"] = SMOKE_WARNING
    return record


def _phase4b_metadata(
    row: dict[str, str],
    class_name: str | None,
    image_provenance: str,
) -> dict[str, Any]:
    evidence_available = _parse_boolish(row.get("evidence_available")) or bool(
        row.get("cam_uri") or row.get("bbox_normalized")
    )
    rsna_localization = _has_rsna_localization(row, image_provenance)
    if class_name == "Pneumonia" and rsna_localization:
        evidence_available = True
    return {
        "evidence_available": evidence_available,
        "evidence_class": class_name if evidence_available else None,
        "rsna_localization": rsna_localization,
        "supervision_quality": "weak",
        "source_phase": "4B",
    }


def _has_rsna_localization(row: dict[str, str], _image_provenance: str) -> bool:
    if row.get("bbox_normalized") or row.get("cam_uri"):
        return True
    if all(row.get(key) not in {None, ""} for key in ["x", "y", "width", "height"]):
        return True
    box_count = row.get("box_count")
    if box_count in {None, ""}:
        return False
    try:
        return int(float(box_count)) > 0
    except ValueError:
        return False


def _quadrant_from_row(row: dict[str, str]) -> str:
    quadrant = row.get("quadrant")
    if quadrant:
        return quadrant
    try:
        x = float(row.get("x", row.get("x_center", "0.5")))
        y = float(row.get("y", row.get("y_center", "0.5")))
        width = float(row.get("width", "0"))
        height = float(row.get("height", "0"))
        image_width = float(row.get("image_width", row.get("width_px", "1")))
        image_height = float(row.get("image_height", row.get("height_px", "1")))
    except ValueError:
        return "central"
    cx = (x + width / 2.0) / max(image_width, 1.0)
    cy = (y + height / 2.0) / max(image_height, 1.0)
    if 0.4 <= cx <= 0.6 and 0.4 <= cy <= 0.6:
        return "central"
    vertical = "upper" if cy < 0.5 else "lower"
    horizontal = "left" if cx < 0.5 else "right"
    return f"{vertical}-{horizontal}"


def _parse_boolish(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _smoke_probabilities(row_index: int, num_classes: int) -> np.ndarray:
    probs = np.full(num_classes, 0.10, dtype=np.float64)
    probs[0] = 0.82
    if num_classes > 1:
        probs[1] = 0.50
    if row_index % 2 == 1 and num_classes > 7:
        probs[7] = 0.78
    return probs


def _artifact_version(path: str | Path) -> str:
    artifact = Path(path)
    if not artifact.exists():
        return f"{artifact.name}@unavailable"
    return f"{artifact.name}@{hashlib.sha256(artifact.read_bytes()).hexdigest()[:12]}"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text()) or {}


def _write_generation_summary(
    config: dict[str, Any],
    output_base: Path,
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    summary_path = output_base.with_suffix(".summary.json")
    summary = {
        "phase": PHASE,
        "config_project": config.get("project", {}).get("name", "medguard-cxr"),
        "output_stem": str(output_base.with_suffix("")),
        "split_counts": {split: len(records) for split, records in splits.items()},
        "WARNING_DO_NOT_USE": SMOKE_WARNING,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vlm_lora.yaml")
    parser.add_argument("--baseline-config", default="configs/baseline_nih.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/baseline_nih_best.pt")
    parser.add_argument("--calibrator", default="calibrators/nih_temp_scaling.pkl")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output", default="data/vqa/synthetic_qa.jsonl")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-distractors", type=int, default=2)
    parser.add_argument("--include-uncertain", type=_parse_bool, default=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
