"""Phase 4B classifier/VLM consistency tests."""

from medguard.api.schemas import SAFETY_DISCLAIMER, VQAResponse
from medguard.models.vlm import check_consistency, vlm_response_or_fallback
from medguard.safety.abstention import PredictionRecord
from medguard.vqa.rule_based import build_default_provenance

NEGATIVE_PNEUMONIA_ANSWER = (
    "The model does not detect visible evidence of Pneumonia above the confidence threshold."
)


def _decision(prediction: int | None, abstained: bool = False) -> PredictionRecord:
    return PredictionRecord(
        image_id="case",
        class_name="Pneumonia",
        prediction=prediction,
        confidence=0.82 if prediction == 1 else 0.1,
        abstained=abstained,
        reason="low_confidence_band" if abstained else "",
    )


def _payload(answer: str) -> dict[str, object]:
    return {
        "question": "Is there evidence of Pneumonia?",
        "answer": answer,
        "confidence": 0.99,
        "evidence": None,
        "abstained": False,
        "reason": "",
        "safety_disclaimer": SAFETY_DISCLAIMER,
        "source": "vlm_zero_shot",
    }


def _fallback() -> VQAResponse:
    return VQAResponse(
        question="Is there evidence of Pneumonia?",
        answer=NEGATIVE_PNEUMONIA_ANSWER,
        confidence=0.1,
        evidence=None,
        abstained=False,
        reason="",
        safety_disclaimer=SAFETY_DISCLAIMER,
        model_provenance=build_default_provenance(),
        source="rule_based",
    )


def test_positive_vlm_answer_is_consistent_with_positive_classifier() -> None:
    assert check_consistency("The model predicts findings consistent with Pneumonia.", _decision(1))


def test_positive_vlm_answer_conflicts_with_negative_classifier() -> None:
    answer = "The model predicts findings consistent with Pneumonia."

    assert not check_consistency(answer, _decision(0))


def test_conflicting_vlm_output_falls_back_to_rule_based() -> None:
    response, result = vlm_response_or_fallback(
        raw_output=_payload("The model predicts findings consistent with Pneumonia."),
        question="Is there evidence of Pneumonia?",
        classifier_decision=_decision(0),
        provenance=build_default_provenance(),
        evidence=None,
        fallback=_fallback(),
        source="vlm_zero_shot",
    )

    assert result.passed is False
    assert result.reason == "classifier_conflict"
    assert response.source == "rule_based"
    assert response.reason == "vlm_output_rejected"
