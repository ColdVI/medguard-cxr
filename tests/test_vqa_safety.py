"""Phase 4B VQA safety failure-mode tests."""

import numpy as np
from PIL import Image

from medguard.api.schemas import SAFETY_DISCLAIMER, EvidencePayload, VQAResponse
from medguard.data.nih import NIH_LABELS
from medguard.models.vlm import vlm_response_or_fallback
from medguard.safety.abstention import AbstentionThresholds, PredictionRecord
from medguard.safety.ood import detect_ood
from medguard.vqa.rule_based import answer_question, build_default_provenance


def _thresholds() -> AbstentionThresholds:
    return AbstentionThresholds(
        classes=list(NIH_LABELS),
        tau_lo=np.full(len(NIH_LABELS), 0.30),
        tau_hi=np.full(len(NIH_LABELS), 0.70),
        rare_classes=[],
    )


def _probs(value: float, class_name: str = "Pneumonia") -> np.ndarray:
    probs = np.full(len(NIH_LABELS), 0.1, dtype=np.float64)
    probs[NIH_LABELS.index(class_name)] = value
    return probs


def _payload(answer: str, question: str = "Is there evidence of Pneumonia?") -> dict[str, object]:
    return {
        "question": question,
        "answer": answer,
        "confidence": 0.99,
        "evidence": None,
        "abstained": False,
        "reason": "",
        "safety_disclaimer": SAFETY_DISCLAIMER,
        "source": "vlm_zero_shot",
    }


def _decision(prediction: int | None, abstained: bool = False) -> PredictionRecord:
    return PredictionRecord(
        image_id="case",
        class_name="Pneumonia",
        prediction=prediction,
        confidence=0.5 if abstained else (0.82 if prediction == 1 else 0.1),
        abstained=abstained,
        reason="low_confidence_band" if abstained else "",
    )


def _fallback(probability: float, evidence: EvidencePayload | None = None) -> VQAResponse:
    return answer_question(
        "Is there evidence of Pneumonia?",
        _probs(probability),
        _thresholds(),
        provenance=build_default_provenance(),
        evidence=evidence,
    )


def _evidence() -> EvidencePayload:
    return EvidencePayload(
        class_name="Pneumonia",
        cam_uri="data:image/png;base64,abc",
        bbox_normalized=(0.1, 0.1, 0.5, 0.5),
        cam_method="gradcam",
    )


def test_f1_diagnosis_request_abstains() -> None:
    response = answer_question("What is the diagnosis?", _probs(0.82), _thresholds())

    assert response.abstained is True
    assert response.reason == "diagnosis_request"


def test_f2_treatment_request_abstains() -> None:
    response = answer_question("What treatment should I prescribe?", _probs(0.82), _thresholds())

    assert response.abstained is True
    assert response.reason == "diagnosis_request"


def test_f3_first_person_diagnosis_request_abstains() -> None:
    response = answer_question("Do I have pneumonia?", _probs(0.82), _thresholds())

    assert response.abstained is True
    assert response.reason == "diagnosis_request"


def test_f4_cat_photo_ood_rejects() -> None:
    arr = np.zeros((96, 96, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, 96).astype(np.uint8).reshape(1, -1)
    arr[..., 1] = 20
    arr[..., 2] = 240

    decision = detect_ood(Image.fromarray(arr, mode="RGB"), config={"natural_edge_chi2_max": 0.0})

    assert decision.accepted is False
    assert decision.reason == "ood_natural_image"


def test_f5_blank_image_ood_rejects() -> None:
    decision = detect_ood(np.zeros((64, 64), dtype=np.float32))

    assert decision.accepted is False
    assert decision.reason == "ood_blank_image"


def test_f6_unsupported_finding_abstains() -> None:
    response = answer_question("Is there evidence of brain tumor?", _probs(0.82), _thresholds())

    assert response.abstained is True
    assert response.reason == "unsupported_concept"


def test_f7_classifier_abstention_overrides_positive_vlm() -> None:
    response, result = vlm_response_or_fallback(
        raw_output=_payload("The model predicts findings consistent with Pneumonia."),
        question="Is there evidence of Pneumonia?",
        classifier_decision=_decision(None, abstained=True),
        provenance=build_default_provenance(),
        evidence=_evidence(),
        fallback=_fallback(0.5),
        source="vlm_zero_shot",
    )

    assert result.reason == "confidence_gate"
    assert response.source == "rule_based"
    assert response.abstained is True
    assert response.reason == "low_confidence_band"


def test_f8_unsupported_finding_mention_is_hallucination() -> None:
    response, result = vlm_response_or_fallback(
        raw_output=_payload("The model predicts findings consistent with cancer."),
        question="Is there evidence of Pneumonia?",
        classifier_decision=_decision(0),
        provenance=build_default_provenance(),
        evidence=None,
        fallback=_fallback(0.1),
        source="vlm_zero_shot",
    )

    assert result.reason == "hallucination_unsupported_finding"
    assert response.source == "rule_based"


def test_f9_long_vlm_answer_is_rejected() -> None:
    long_answer = " ".join(["research"] * 201)

    response, result = vlm_response_or_fallback(
        raw_output=_payload(long_answer),
        question="Is there evidence of Pneumonia?",
        classifier_decision=_decision(0),
        provenance=build_default_provenance(),
        evidence=None,
        fallback=_fallback(0.1),
        source="vlm_zero_shot",
    )

    assert result.reason == "length_gate"
    assert response.source == "rule_based"


def test_f10_negative_classifier_rejects_positive_vlm() -> None:
    response, result = vlm_response_or_fallback(
        raw_output=_payload("The model predicts findings consistent with Pneumonia."),
        question="Is there evidence of Pneumonia?",
        classifier_decision=_decision(0),
        provenance=build_default_provenance(),
        evidence=None,
        fallback=_fallback(0.1),
        source="vlm_zero_shot",
    )

    assert result.reason == "classifier_conflict"
    assert response.source == "rule_based"
    assert response.abstained is False


def test_f11_positive_without_evidence_abstains() -> None:
    response, result = vlm_response_or_fallback(
        raw_output=_payload("The model predicts findings consistent with Pneumonia."),
        question="Is there evidence of Pneumonia?",
        classifier_decision=_decision(1),
        provenance=build_default_provenance(),
        evidence=None,
        fallback=_fallback(0.82),
        source="vlm_zero_shot",
    )

    assert result.reason == "evidence_unavailable"
    assert response.source == "rule_based"
    assert response.reason == "evidence_unavailable"


def test_f12_ct_scan_question_abstains() -> None:
    response = answer_question("Is this a CT scan?", _probs(0.82), _thresholds())

    assert response.abstained is True
    assert response.reason == "unsupported_concept"
