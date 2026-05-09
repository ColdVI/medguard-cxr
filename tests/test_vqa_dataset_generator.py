"""Synthetic VQA dataset generator tests."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scripts.generate_vqa_dataset import (
    generate_dataset_records,
    patient_disjoint_split,
    read_manifest,
    validate_jsonl_provenance,
    write_split_jsonl,
)

from medguard.data.nih import NIH_LABELS
from medguard.safety.abstention import AbstentionThresholds
from medguard.vqa.templates import validate_qa_record


def _thresholds() -> AbstentionThresholds:
    return AbstentionThresholds(
        classes=list(NIH_LABELS),
        tau_lo=np.full(len(NIH_LABELS), 0.30),
        tau_hi=np.full(len(NIH_LABELS), 0.70),
        rare_classes=[],
    )


def _manifest() -> list[dict[str, str]]:
    return [
        {"image_id": f"patient{i}_000.png", "path": f"data/rsna/image{i}.png"}
        for i in range(5)
    ]


def test_generator_produces_positive_negative_uncertain_unsupported_diagnosis() -> None:
    records = generate_dataset_records(_manifest()[:1], _thresholds(), n_distractors=2)
    kinds = {record["label_kind"] for record in records}

    assert {
        "positive",
        "negative",
        "uncertain",
        "unsupported_concept",
        "diagnosis_request",
    } <= kinds


def test_generator_refuses_clinical_tokens_in_answers() -> None:
    with pytest.raises(ValueError, match="banned clinical token"):
        validate_qa_record({"answer": "This is definitely a diagnosis."})


def test_generator_is_patient_disjoint_across_splits() -> None:
    records = generate_dataset_records(_manifest(), _thresholds(), n_distractors=1)
    splits = patient_disjoint_split(records)
    patient_sets = {
        split: {record["patient_id"] for record in split_records}
        for split, split_records in splits.items()
    }

    assert patient_sets["train"].isdisjoint(patient_sets["val"])
    assert patient_sets["train"].isdisjoint(patient_sets["test"])
    assert patient_sets["val"].isdisjoint(patient_sets["test"])


def test_generator_records_image_provenance() -> None:
    records = generate_dataset_records(_manifest()[:1], _thresholds())

    assert all(record["image_provenance"] == "synthetic_smoke" for record in records)


def test_generator_emits_warning_when_smoke_classifier_used() -> None:
    records = generate_dataset_records(_manifest()[:1], _thresholds(), is_smoke=True)

    assert all(
        record["WARNING_DO_NOT_USE"] == "synthetic_smoke_only_not_a_real_evaluation"
        for record in records
    )


def test_manifest_reader_requires_image_id_and_path(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "path"])
        writer.writeheader()
        writer.writerow({"image_id": "p1_0.png", "path": "data/rsna/p1_0.png"})

    assert read_manifest(path)[0]["image_id"] == "p1_0.png"


def test_generator_refuses_mixed_provenance_for_training_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    records = [
        {"image_provenance": "nih"},
        {"image_provenance": "rsna"},
    ]
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="must not mix"):
        validate_jsonl_provenance([path])


def test_write_split_jsonl_outputs_three_files(tmp_path: Path) -> None:
    records = generate_dataset_records(_manifest(), _thresholds(), n_distractors=1)
    splits = patient_disjoint_split(records)

    write_split_jsonl(splits, tmp_path / "synthetic_qa.jsonl")

    assert (tmp_path / "synthetic_qa.train.jsonl").exists()
    assert (tmp_path / "synthetic_qa.val.jsonl").exists()
    assert (tmp_path / "synthetic_qa.test.jsonl").exists()
