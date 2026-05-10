"""Classifier-grounded, rule-based VQA for Phase 4A."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from medguard.api.schemas import (
    SAFETY_DISCLAIMER,
    SMOKE_WARNING,
    EvidencePayload,
    ModelProvenance,
    VQAResponse,
)
from medguard.data.nih import NIH_LABELS
from medguard.safety.abstention import AbstentionThresholds, apply_abstention
from medguard.safety.ood import OODDecision
from medguard.safety.question_filter import (
    classify_question,
    extract_supported_finding,
    requested_finding_text,
)
from medguard.vqa.templates import (
    DIAGNOSIS_REQUEST_ANSWER,
    OOD_REJECTED_ANSWER,
    UNSUPPORTED_CONCEPT_ANSWER,
    answer_for_label_kind,
)

PHASE = "4"


def is_available() -> bool:
    """Return whether rule-based VQA is implemented."""

    return True


def build_default_provenance(
    classifier_checkpoint_sha256: str = "unavailable",
    calibrator_sha256: str | None = None,
    is_smoke: bool = True,
) -> ModelProvenance:
    """Build a conservative provenance block for smoke-first Phase 4A."""

    return ModelProvenance(
        classifier_checkpoint_sha256=classifier_checkpoint_sha256,
        calibrator_sha256=calibrator_sha256,
        is_smoke=is_smoke,
        warning=SMOKE_WARNING if is_smoke else None,
    )


def answer_question(
    question: str,
    probabilities: np.ndarray,
    thresholds: AbstentionThresholds,
    provenance: ModelProvenance | None = None,
    ood_decision: OODDecision | None = None,
    evidence: EvidencePayload | None = None,
    require_evidence_for_positive: bool = True,
) -> VQAResponse:
    """Answer a supported finding question using only fixed templates and Phase 2 gates."""

    model_provenance = provenance or build_default_provenance()
    if ood_decision is not None and (not ood_decision.accepted or ood_decision.warning_only):
        return _abstain(question, "ood_rejected", ood_decision.reason, model_provenance)

    requested_finding = requested_finding_text(question)
    if requested_finding is not None and extract_supported_finding(
        question,
        class_names=thresholds.classes,
    ) is None:
        return VQAResponse(
            question=question,
            answer=UNSUPPORTED_CONCEPT_ANSWER.format(n_findings=len(NIH_LABELS)),
            confidence=0.0,
            evidence=None,
            abstained=True,
            reason="unsupported_concept",
            safety_disclaimer=SAFETY_DISCLAIMER,
            model_provenance=model_provenance,
            source="rule_based",
        )

    kind = classify_question(question, class_names=thresholds.classes)
    if kind == "diagnosis_request":
        return _abstain(question, "diagnosis_request", "diagnosis_request", model_provenance)
    if kind == "unsupported_concept":
        return _abstain(question, "unsupported_concept", "unsupported_concept", model_provenance)
    if kind == "unparseable":
        return _abstain(question, "unsupported_concept", "unsupported_concept", model_provenance)

    class_name = extract_supported_finding(question, class_names=thresholds.classes)
    if class_name is None:
        return _abstain(question, "unsupported_concept", "unsupported_concept", model_provenance)

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    records = apply_abstention(probs, thresholds)[0]
    class_index = thresholds.classes.index(class_name)
    record = records[class_index]
    if record.abstained:
        return VQAResponse(
            question=question,
            answer=answer_for_label_kind("uncertain", class_name),
            confidence=0.0,
            evidence=None,
            abstained=True,
            reason="low_confidence_band",
            safety_disclaimer=SAFETY_DISCLAIMER,
            model_provenance=model_provenance,
            source="rule_based",
        )
    if record.prediction == 1 and require_evidence_for_positive and evidence is None:
        return VQAResponse(
            question=question,
            answer=answer_for_label_kind("uncertain", class_name),
            confidence=0.0,
            evidence=None,
            abstained=True,
            reason="evidence_unavailable",
            safety_disclaimer=SAFETY_DISCLAIMER,
            model_provenance=model_provenance,
            source="rule_based",
        )

    label_kind = "positive" if record.prediction == 1 else "negative"
    return VQAResponse(
        question=question,
        answer=answer_for_label_kind(label_kind, class_name),
        confidence=float(record.confidence),
        evidence=evidence if record.prediction == 1 else None,
        abstained=False,
        reason="",
        safety_disclaimer=SAFETY_DISCLAIMER,
        model_provenance=model_provenance,
        source="rule_based",
    )


def answer_from_prediction_record(
    prediction: int | None,
    abstained: bool,
    class_name: str,
) -> str:
    """Return the fixed answer template for one prediction state."""

    if abstained or prediction is None:
        return answer_for_label_kind("uncertain", class_name)
    return answer_for_label_kind("positive" if prediction == 1 else "negative", class_name)


def thresholds_config_with_classes(config: Mapping[str, Any]) -> dict[str, Any]:
    """Attach NIH class order when a calibration YAML omits it."""

    merged = dict(config)
    merged.setdefault("classes", list(NIH_LABELS))
    return merged


def _abstain(
    question: str,
    answer_kind: str,
    reason: str,
    provenance: ModelProvenance,
) -> VQAResponse:
    if answer_kind == "diagnosis_request":
        answer = DIAGNOSIS_REQUEST_ANSWER
    elif answer_kind == "ood_rejected":
        answer = OOD_REJECTED_ANSWER.format(reason=reason)
    elif answer_kind == "unsupported_concept":
        answer = UNSUPPORTED_CONCEPT_ANSWER.format(n_findings=len(NIH_LABELS))
    else:
        answer = answer_for_label_kind("uncertain", answer_kind)
    return VQAResponse(
        question=question,
        answer=answer,
        confidence=0.0,
        evidence=None,
        abstained=True,
        reason=reason,  # type: ignore[arg-type]
        safety_disclaimer=SAFETY_DISCLAIMER,
        model_provenance=provenance,
        source="rule_based",
    )
