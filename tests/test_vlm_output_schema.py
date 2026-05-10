"""Phase 4B VLM output schema validation tests."""

import pytest

from medguard.api.schemas import SAFETY_DISCLAIMER, VQAResponse
from medguard.models.vlm import parse_vlm_output, validate_vlm_payload
from medguard.vqa.rule_based import build_default_provenance

NEGATIVE_PNEUMONIA_ANSWER = (
    "The model does not detect visible evidence of Pneumonia above the confidence threshold."
)


def _payload(source: str = "vlm_zero_shot") -> dict[str, object]:
    return {
        "question": "Is there evidence of Pneumonia?",
        "answer": NEGATIVE_PNEUMONIA_ANSWER,
        "confidence": 0.1,
        "evidence": None,
        "abstained": False,
        "reason": "",
        "safety_disclaimer": SAFETY_DISCLAIMER,
        "source": source,
    }


def test_vqa_response_accepts_source_field() -> None:
    response = VQAResponse(
        question="Is there evidence of Pneumonia?",
        answer=NEGATIVE_PNEUMONIA_ANSWER,
        confidence=0.1,
        evidence=None,
        abstained=False,
        reason="",
        safety_disclaimer=SAFETY_DISCLAIMER,
        model_provenance=build_default_provenance(),
        source="vlm_zero_shot",
    )

    assert response.source == "vlm_zero_shot"


def test_validate_vlm_payload_requires_all_fields() -> None:
    payload = _payload()

    parsed = validate_vlm_payload(payload, expected_question=str(payload["question"]))

    assert parsed["source"] == "vlm_zero_shot"


def test_validate_vlm_payload_rejects_missing_source() -> None:
    payload = _payload()
    payload.pop("source")

    with pytest.raises(ValueError, match="missing required fields"):
        validate_vlm_payload(payload)


def test_parse_vlm_output_accepts_fenced_json() -> None:
    parsed = parse_vlm_output('```json\n{"answer": "ok"}\n```')

    assert parsed == {"answer": "ok"}


def test_parse_vlm_output_rejects_free_text() -> None:
    with pytest.raises(ValueError, match="structured JSON"):
        parse_vlm_output("The model sees something.")
