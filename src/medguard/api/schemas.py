"""Phase 4 API schemas and safety constants."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PHASE = "4"

SAFETY_DISCLAIMER = (
    "MedGuard-CXR is a research tool only. It is not a clinical diagnostic system, "
    "must not be used for patient care, and does not provide medical advice."
)
SMOKE_WARNING = "synthetic_smoke_only_not_a_real_evaluation"

REASON_CODES = {
    "": "no abstention",
    "low_confidence_band": "calibrated probability sits in the abstention band",
    "ood_natural_image": "input was not classified as a chest X-ray",
    "ood_blank_image": "input image is blank or near-blank",
    "ood_corrupted_image": "input image appears corrupted or is noise-only",
    "ood_unsupported_view": "input appears to be a non-PA frontal view",
    "diagnosis_request": "question requests diagnosis, treatment, or medical advice",
    "unsupported_concept": "question is outside the scope of trained findings",
    "unsupported_finding": "requested finding is not in the trained class set",
    "evidence_unavailable": "C4 violation prevented: positive result without retrievable evidence",
    "vlm_output_rejected": "experimental VLM output failed the template safety filter",
}

ReasonCode = Literal[
    "",
    "low_confidence_band",
    "ood_natural_image",
    "ood_blank_image",
    "ood_corrupted_image",
    "ood_unsupported_view",
    "diagnosis_request",
    "unsupported_concept",
    "unsupported_finding",
    "evidence_unavailable",
    "vlm_output_rejected",
]
VQASource = Literal["rule_based", "vlm_zero_shot", "vlm_lora"]


def is_available() -> bool:
    """Return whether Phase 4 API schemas are implemented."""

    return True


class EvidencePayload(BaseModel):
    """Visual evidence metadata attached to non-abstained positive VQA answers."""

    class_name: str
    cam_uri: str | None
    bbox_normalized: tuple[float, float, float, float] | None
    cam_method: Literal["gradcam", "gradcam_plus_plus"]

    @field_validator("bbox_normalized")
    @classmethod
    def _validate_bbox(
        cls,
        value: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return value
        if any(coord < 0.0 or coord > 1.0 for coord in value):
            raise ValueError("bbox_normalized coordinates must be in [0, 1].")
        x0, y0, x1, y1 = value
        if x0 > x1 or y0 > y1:
            raise ValueError("bbox_normalized must be xyxy ordered.")
        return value


class ModelProvenance(BaseModel):
    """Provenance for every Phase 4 output."""

    classifier_checkpoint_sha256: str
    calibrator_sha256: str | None
    is_smoke: bool
    warning: str | None

    @field_validator("warning")
    @classmethod
    def _validate_warning(cls, value: str | None, info: object) -> str | None:
        data = getattr(info, "data", {})
        if data.get("is_smoke"):
            if value != SMOKE_WARNING:
                raise ValueError("Smoke provenance must carry the canonical warning.")
        elif value is not None:
            raise ValueError("Non-smoke provenance must not carry a warning.")
        return value


class VQAResponse(BaseModel):
    """Safety-aware VQA response contract."""

    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: EvidencePayload | None
    abstained: bool
    reason: ReasonCode
    safety_disclaimer: str
    model_provenance: ModelProvenance
    source: VQASource = "rule_based"

    @field_validator("safety_disclaimer")
    @classmethod
    def _validate_disclaimer(cls, value: str) -> str:
        if value != SAFETY_DISCLAIMER:
            raise ValueError("safety_disclaimer must be the canonical non-clinical disclaimer.")
        return value

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        if value not in REASON_CODES:
            raise ValueError(f"Unknown reason code: {value}")
        return value


class PredictionPayload(BaseModel):
    """One class prediction returned by ``/predict``."""

    class_name: str
    prediction: int | None
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool
    reason: ReasonCode


class OODPayload(BaseModel):
    """Serialized OOD decision."""

    accepted: bool
    reason: ReasonCode
    score: dict[str, float]
    warning_only: bool = False


class PredictResponse(BaseModel):
    """Response body for ``POST /predict``."""

    predictions: list[PredictionPayload] | None
    ood: OODPayload
    model_provenance: ModelProvenance
    safety_disclaimer: str = SAFETY_DISCLAIMER


class ExplainResponse(BaseModel):
    """Response body for ``POST /explain``."""

    evidence: EvidencePayload | None
    reason: ReasonCode = ""
    model_provenance: ModelProvenance
    safety_disclaimer: str = SAFETY_DISCLAIMER


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: Literal["ok"]
    phase: Literal[4]
    components: dict[str, Literal["loaded", "unavailable", "disabled"]]
    model_provenance: ModelProvenance
    safety_disclaimer: str = SAFETY_DISCLAIMER


class ProblemDetails(BaseModel):
    """RFC 7807-style error body with MedGuard safety fields."""

    model_config = ConfigDict(populate_by_name=True)

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    reason: ReasonCode = ""
    safety_disclaimer: str = SAFETY_DISCLAIMER
    model_provenance: ModelProvenance
