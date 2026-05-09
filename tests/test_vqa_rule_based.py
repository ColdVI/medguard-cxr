"""Rule-based VQA response tests."""

import numpy as np

from medguard.api.schemas import SAFETY_DISCLAIMER, SMOKE_WARNING, EvidencePayload
from medguard.data.nih import NIH_LABELS
from medguard.safety.abstention import AbstentionThresholds
from medguard.safety.ood import OODDecision
from medguard.vqa.rule_based import answer_question, build_default_provenance


def _thresholds() -> AbstentionThresholds:
    return AbstentionThresholds(
        classes=list(NIH_LABELS),
        tau_lo=np.full(len(NIH_LABELS), 0.30),
        tau_hi=np.full(len(NIH_LABELS), 0.70),
        rare_classes=[],
    )


def _probs(value: float = 0.1, class_name: str = "Pneumothorax") -> np.ndarray:
    probs = np.full(len(NIH_LABELS), value, dtype=np.float64)
    probs[NIH_LABELS.index(class_name)] = value
    return probs


def _evidence(class_name: str = "Pneumothorax") -> EvidencePayload:
    return EvidencePayload(
        class_name=class_name,
        cam_uri="data:image/png;base64,abc",
        bbox_normalized=(0.1, 0.1, 0.5, 0.5),
        cam_method="gradcam",
    )


def test_supported_finding_returns_structured_answer() -> None:
    probs = _probs(0.82)

    response = answer_question(
        "Is there evidence of Pneumothorax?",
        probs,
        _thresholds(),
        evidence=_evidence(),
    )

    assert response.abstained is False
    assert response.confidence == 0.82
    assert response.evidence is not None
    assert response.safety_disclaimer == SAFETY_DISCLAIMER


def test_low_confidence_vqa_abstains() -> None:
    response = answer_question("Is there evidence of Pneumothorax?", _probs(0.5), _thresholds())

    assert response.abstained is True
    assert response.reason == "low_confidence_band"
    assert response.confidence == 0.0


def test_diagnosis_question_is_rejected() -> None:
    response = answer_question("What is the diagnosis?", _probs(0.82), _thresholds())

    assert response.abstained is True
    assert response.reason == "diagnosis_request"


def test_unsupported_finding_is_rejected() -> None:
    response = answer_question("Is there evidence of Brain Tumor?", _probs(0.82), _thresholds())

    assert response.abstained is True
    assert response.reason == "unsupported_finding"


def test_safety_disclaimer_always_present() -> None:
    cases = [
        ("Is there evidence of Pneumothorax?", _probs(0.1), None),
        ("Is there evidence of Pneumothorax?", _probs(0.5), None),
        ("What treatment should I prescribe?", _probs(0.1), None),
        ("Does this show kidney stones?", _probs(0.1), None),
        ("Is there evidence of Brain Tumor?", _probs(0.1), None),
    ]
    for question, probs, evidence in cases:
        response = answer_question(question, probs, _thresholds(), evidence=evidence)
        assert response.safety_disclaimer == SAFETY_DISCLAIMER


def test_confidence_zeroed_on_abstention() -> None:
    response = answer_question(
        "Is there evidence of Pneumothorax?",
        _probs(0.82),
        _thresholds(),
        evidence=None,
    )

    assert response.reason == "evidence_unavailable"
    assert response.confidence == 0.0


def test_response_contains_model_provenance() -> None:
    response = answer_question("Is there evidence of Pneumothorax?", _probs(0.1), _thresholds())

    assert response.model_provenance.classifier_checkpoint_sha256


def test_smoke_mode_sets_warning_field() -> None:
    provenance = build_default_provenance(is_smoke=True)

    response = answer_question(
        "Is there evidence of Pneumothorax?",
        _probs(0.1),
        _thresholds(),
        provenance=provenance,
    )

    assert response.model_provenance.warning == SMOKE_WARNING


def test_ood_gate_short_circuits_vqa() -> None:
    response = answer_question(
        "Is there evidence of Pneumothorax?",
        _probs(0.82),
        _thresholds(),
        ood_decision=OODDecision(False, "ood_blank_image", {"std": 0.0}),
    )

    assert response.abstained is True
    assert response.reason == "ood_blank_image"
    assert "safety gate" in response.answer
