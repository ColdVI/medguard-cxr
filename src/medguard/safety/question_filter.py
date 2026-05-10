"""Deterministic question classifier for Phase 4 VQA safety gates."""

from __future__ import annotations

import re
from typing import Literal

from medguard.data.nih import NIH_LABELS

PHASE = "4"

QuestionKind = Literal[
    "supported_finding_query",
    "diagnosis_request",
    "unsupported_concept",
    "unparseable",
]

DIAGNOSIS_REQUEST_PATTERN = re.compile(
    r"\b("
    r"diagnosis|diagnose|treatment|prescrib|admit|recommend|prognos|advise|"
    r"should I|do I have"
    r")\b",
    flags=re.IGNORECASE,
)
UNSUPPORTED_CONCEPT_PATTERN = re.compile(
    r"\b(CT|MRI|ultrasound|abdomen|kidney|fracture|bone scan|mammogram|hip|knee|skull)\b",
    flags=re.IGNORECASE,
)
SUPPORTED_FINDING_PATTERN = re.compile(
    r"^Is there evidence of (?P<finding>[A-Za-z_ ]+)\?(?:\s*Where\?)?$",
    flags=re.IGNORECASE,
)
CONFIDENCE_FINDING_PATTERN = re.compile(
    r"^How confident is the model about (?P<finding>[A-Za-z_ ]+)\?$",
    flags=re.IGNORECASE,
)


def is_available() -> bool:
    """Return whether the Phase 4 question filter is implemented."""

    return True


def classify_question(
    question: str,
    class_names: list[str] | None = None,
) -> QuestionKind:
    """Classify a VQA question without calling an LLM."""

    if DIAGNOSIS_REQUEST_PATTERN.search(question):
        return "diagnosis_request"
    if UNSUPPORTED_CONCEPT_PATTERN.search(question):
        return "unsupported_concept"
    finding = extract_supported_finding(question, class_names=class_names)
    if finding is not None:
        return "supported_finding_query"
    return "unparseable"


def extract_supported_finding(
    question: str,
    class_names: list[str] | None = None,
) -> str | None:
    """Return the canonical class named in a supported finding question."""

    match = SUPPORTED_FINDING_PATTERN.match(question.strip())
    if match is None:
        match = CONFIDENCE_FINDING_PATTERN.match(question.strip())
    if match is None:
        return None
    requested = _normalize_finding(match.group("finding"))
    for class_name in class_names or NIH_LABELS:
        if requested == _normalize_finding(class_name):
            return class_name
    return None


def requested_finding_text(question: str) -> str | None:
    """Return the finding-like text from the canonical question form, if present."""

    match = SUPPORTED_FINDING_PATTERN.match(question.strip())
    if match is None:
        match = CONFIDENCE_FINDING_PATTERN.match(question.strip())
    if match is None:
        return None
    return match.group("finding").strip()


def _normalize_finding(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").strip().lower())
