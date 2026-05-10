"""Closed answer templates for Phase 4 rule-based VQA."""

from __future__ import annotations

import re
from typing import Any

from medguard.api.schemas import SMOKE_WARNING

PHASE = "4"

POSITIVE_ANSWER = "The model predicts findings consistent with {finding}."
NEGATIVE_ANSWER = (
    "The model does not detect visible evidence of {finding} above the confidence threshold."
)
UNCERTAIN_ANSWER = "The model is not confident enough to answer about {finding} for this image."
LOCALIZATION_ANSWER = (
    "The model predicts findings consistent with {finding}. A region of interest was "
    "identified in the {quadrant} of the image."
)
CONFIDENCE_ANSWER = (
    "The calibrated classifier probability for {finding} is {confidence:.2f}. "
    "This is a research signal only."
)
MULTI_FINDING_ANSWER = "The model predicts findings consistent with {findings}."
NO_MULTI_FINDING_ANSWER = "The model does not detect visible evidence of supported findings."
UNSUPPORTED_CONCEPT_ANSWER = (
    "This question is outside the scope of MedGuard-CXR. The system only reasons about "
    "the {n_findings} findings it was trained on, on chest X-rays."
)
DIAGNOSIS_REQUEST_ANSWER = (
    "MedGuard-CXR does not provide diagnoses, treatment, or clinical advice. "
    "It is a research tool only."
)
OOD_REJECTED_ANSWER = (
    "The submitted image was not accepted by the safety gate ({reason}); "
    "MedGuard-CXR cannot answer questions about it."
)

UNSUPPORTED_CONCEPT_QUESTIONS = [
    "Is this a CT scan?",
    "Does this show a fractured femur?",
    "Is this an abdominal X-ray?",
    "Does this show kidney stones?",
]
DIAGNOSIS_REQUEST_QUESTIONS = [
    "What is the diagnosis?",
    "What treatment should I prescribe?",
    "Is this patient critically ill?",
    "Should I admit this patient?",
]

BANNED_ANSWER_TOKENS = [
    "diagnosis",
    "diagnose",
    "treatment",
    "prescribe",
    "recommend",
    "medical advice",
    "prognosis",
    "definitely",
    "likely",
    "probably has",
]


def is_available() -> bool:
    """Return whether Phase 4 VQA templates are implemented."""

    return True


def display_finding(class_name: str) -> str:
    """Render an NIH label for user-facing fixed templates."""

    return class_name.replace("_", " ")


def answer_for_label_kind(
    label_kind: str,
    finding: str | None = None,
    n_findings: int = 14,
) -> str:
    """Return a fixed safe answer template for a generated QA label kind."""

    display = display_finding(finding or "")
    if label_kind == "positive":
        return POSITIVE_ANSWER.format(finding=display)
    if label_kind == "negative":
        return NEGATIVE_ANSWER.format(finding=display)
    if label_kind == "uncertain":
        return UNCERTAIN_ANSWER.format(finding=display)
    if label_kind == "unsupported_concept":
        return UNSUPPORTED_CONCEPT_ANSWER.format(n_findings=n_findings)
    if label_kind == "diagnosis_request":
        return DIAGNOSIS_REQUEST_ANSWER
    if label_kind == "ood_rejected":
        return OOD_REJECTED_ANSWER.format(reason=finding or "ood_rejected")
    raise ValueError(f"Unsupported QA label kind: {label_kind}")


def localization_answer(finding: str, quadrant: str) -> str:
    """Return the fixed weak-localization answer template."""

    return LOCALIZATION_ANSWER.format(finding=display_finding(finding), quadrant=quadrant)


def confidence_answer(finding: str, confidence: float) -> str:
    """Return the fixed calibrated-confidence answer template."""

    return CONFIDENCE_ANSWER.format(finding=display_finding(finding), confidence=confidence)


def multi_finding_answer(findings: list[str]) -> str:
    """Return the fixed multi-finding answer template."""

    if not findings:
        return NO_MULTI_FINDING_ANSWER
    rendered = ", ".join(display_finding(finding) for finding in findings)
    return MULTI_FINDING_ANSWER.format(findings=rendered)


def validate_qa_record(record: dict[str, Any]) -> None:
    """Raise if a QA record can leak unsafe clinical-tone answer text."""

    answer = str(record.get("answer", ""))
    if answer == DIAGNOSIS_REQUEST_ANSWER:
        return
    lower = answer.lower()
    for token in BANNED_ANSWER_TOKENS:
        if token.lower() in lower:
            raise ValueError(f"QA answer contains banned clinical token: {token}")
    if record.get("model_abstained") and record.get("model_confidence") not in {0, None}:
        raise ValueError("Abstained QA records must not expose underlying model confidence.")
    if record.get("WARNING_DO_NOT_USE") not in {None, SMOKE_WARNING}:
        raise ValueError("Unexpected smoke warning marker.")
    if "source_phase" in record:
        required = {
            "evidence_available",
            "evidence_class",
            "rsna_localization",
            "supervision_quality",
            "source_phase",
        }
        missing = required - set(record)
        if missing:
            raise ValueError(f"QA record missing Phase 4B fields: {sorted(missing)}")
        if record.get("supervision_quality") != "weak":
            raise ValueError("Phase 4B QA supervision_quality must be weak.")
        if record.get("source_phase") != "4B":
            raise ValueError("Phase 4B QA source_phase must be 4B.")


def template_matches(answer: str, finding: str | None = None, n_findings: int = 14) -> bool:
    """Return whether an answer matches one of the closed templates."""

    normalized = _normalize(answer)
    candidates = [
        answer_for_label_kind("unsupported_concept", n_findings=n_findings),
        DIAGNOSIS_REQUEST_ANSWER,
        NO_MULTI_FINDING_ANSWER,
    ]
    if finding:
        candidates.extend(
            [
                answer_for_label_kind("positive", finding=finding),
                answer_for_label_kind("negative", finding=finding),
                answer_for_label_kind("uncertain", finding=finding),
                confidence_answer(finding, 0.0),
            ]
        )
    return any(normalized == _normalize(candidate) for candidate in candidates)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
