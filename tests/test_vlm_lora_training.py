"""Phase 4B QLoRA training helper tests."""

from pathlib import Path

from scripts.train_vlm_lora import (
    _limit_records,
    _target_payload,
    _target_text,
    _training_blocker,
)

from medguard.api.schemas import SAFETY_DISCLAIMER


def _record() -> dict[str, object]:
    return {
        "question": "Is there evidence of Pneumonia?",
        "answer": "The classifier output is consistent with Pneumonia.",
        "label_class": "Pneumonia",
        "label_kind": "positive",
        "model_confidence": 0.82,
        "model_abstained": False,
        "abstention_reason": "",
        "evidence_available": True,
        "evidence_class": "Pneumonia",
    }


def test_target_payload_matches_required_vlm_schema() -> None:
    payload = _target_payload(_record())

    assert payload["question"] == "Is there evidence of Pneumonia?"
    assert payload["source"] == "vlm_lora"
    assert payload["safety_disclaimer"] == SAFETY_DISCLAIMER
    assert payload["confidence"] == 0.82
    assert payload["abstained"] is False
    assert payload["reason"] == ""
    assert payload["evidence"]["class_name"] == "Pneumonia"


def test_target_payload_abstains_for_diagnosis_request() -> None:
    record = _record()
    record["label_kind"] = "diagnosis_request"
    record["answer"] = "I cannot answer diagnosis or treatment questions."

    payload = _target_payload(record)

    assert payload["abstained"] is True
    assert payload["reason"] == "diagnosis_request"


def test_target_text_is_json_completion() -> None:
    target = _target_text(_record())

    assert target.endswith("<|im_end|>")
    assert '"source": "vlm_lora"' in target


def test_limit_records_handles_empty_limit() -> None:
    records = [{"id": 1}, {"id": 2}]

    assert _limit_records(records, None) == records
    assert _limit_records(records, "") == records
    assert _limit_records(records, 1) == [{"id": 1}]


def test_training_blocker_preserves_disabled_default(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text("{}\n")
    val.write_text("{}\n")

    blocker = _training_blocker(
        {"vlm": {"training": {"enabled": False}}},
        train,
        val,
        smoke=False,
    )

    assert blocker == "training_disabled_in_config"
