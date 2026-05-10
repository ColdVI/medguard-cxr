"""FastAPI app for Phase 4 safety-aware MedGuard-CXR inference."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from medguard.api.middleware import AuditLogMiddleware, DisclaimerMiddleware, ProvenanceMiddleware
from medguard.api.schemas import (
    SAFETY_DISCLAIMER,
    SMOKE_WARNING,
    EvidencePayload,
    ExplainResponse,
    HealthResponse,
    ModelProvenance,
    OODPayload,
    PredictionPayload,
    PredictResponse,
    ProblemDetails,
    VQAResponse,
)
from medguard.data.nih import NIH_LABELS
from medguard.models.vlm import VLMInferenceEngine, answer_with_optional_vlm, load_vlm
from medguard.safety.abstention import apply_abstention, load_thresholds_from_config
from medguard.safety.ood import OODDecision, detect_ood, load_ood_config
from medguard.vqa.rule_based import answer_question, thresholds_config_with_classes

PHASE = "4"


class ImageRequest(BaseModel):
    """Image input accepted by prediction and explanation endpoints."""

    image: str | None = None
    image_url: str | None = None
    image_path: str | None = None


class ExplainRequest(ImageRequest):
    """Explanation request body."""

    class_name: str


class VQARequest(ImageRequest):
    """VQA request body."""

    question: str


def is_available() -> bool:
    """Return whether the Phase 4 FastAPI app is implemented."""

    return True


class MedGuardService:
    """Small deterministic service shell for smoke-safe Phase 4A behavior."""

    def __init__(
        self,
        calibration_config: str | Path = "configs/calibration.yaml",
        ood_config: str | Path = "configs/ood.yaml",
        vlm_config: str | Path = "configs/vlm_lora.yaml",
        provenance: ModelProvenance | None = None,
        fixed_probabilities: np.ndarray | None = None,
        vlm_engine: VLMInferenceEngine | None = None,
        enable_vlm: bool | None = None,
    ) -> None:
        self.provenance = provenance or default_model_provenance()
        self.thresholds = load_thresholds_from_config(
            thresholds_config_with_classes(_load_yaml(calibration_config))
        )
        self.ood_config = load_ood_config(ood_config)
        self.vlm_config = _load_yaml(vlm_config)
        self.fixed_probabilities = fixed_probabilities
        self.vlm_engine = vlm_engine or self._maybe_load_vlm(enable_vlm)
        self.last_predictions: dict[str, PredictionPayload] = {}

    def health(self) -> HealthResponse:
        """Return component availability."""

        vlm_status = "disabled"
        if self.vlm_engine is not None:
            vlm_status = "loaded"
        elif self.vlm_config.get("vlm", {}).get("enabled"):
            vlm_status = "unavailable"
        return HealthResponse(
            status="ok",
            phase=4,
            components={
                "classifier": "loaded",
                "calibrator": "loaded",
                "vlm": vlm_status,
                "ood": "loaded",
            },
            model_provenance=self.provenance,
            safety_disclaimer=SAFETY_DISCLAIMER,
        )

    def predict(self, image: Image.Image | np.ndarray) -> PredictResponse:
        """Run OOD and selective prediction."""

        ood = detect_ood(image, config=self.ood_config)
        if not ood.accepted:
            return PredictResponse(
                predictions=None,
                ood=_ood_payload(ood),
                model_provenance=self.provenance,
                safety_disclaimer=SAFETY_DISCLAIMER,
            )
        probs = self._probabilities(image)
        records = apply_abstention(probs.reshape(1, -1), self.thresholds)[0]
        predictions = [
            PredictionPayload(
                class_name=record.class_name,
                prediction=record.prediction,
                confidence=float(record.confidence),
                abstained=record.abstained,
                reason=record.reason,  # type: ignore[arg-type]
            )
            for record in records
        ]
        self.last_predictions = {prediction.class_name: prediction for prediction in predictions}
        return PredictResponse(
            predictions=predictions,
            ood=_ood_payload(ood),
            model_provenance=self.provenance,
            safety_disclaimer=SAFETY_DISCLAIMER,
        )

    def explain(self, image: Image.Image | np.ndarray, class_name: str) -> ExplainResponse:
        """Return smoke evidence only for non-abstained positive decisions."""

        prediction = self.last_predictions.get(class_name)
        if prediction is None:
            self.predict(image)
            prediction = self.last_predictions.get(class_name)
        if prediction is None or prediction.abstained or prediction.prediction != 1:
            return ExplainResponse(
                evidence=None,
                reason="evidence_unavailable",
                model_provenance=self.provenance,
                safety_disclaimer=SAFETY_DISCLAIMER,
            )
        evidence = _smoke_evidence(class_name)
        return ExplainResponse(
            evidence=evidence,
            reason="",
            model_provenance=self.provenance,
            safety_disclaimer=SAFETY_DISCLAIMER,
        )

    def vqa(self, image: Image.Image | np.ndarray, question: str) -> VQAResponse:
        """Answer a VQA question through deterministic gates."""

        ood = detect_ood(image, config=self.ood_config)
        probs = self._probabilities(image)
        evidence = _smoke_evidence(_class_from_question(question)) if ood.accepted else None
        rule_based = answer_question(
            question=question,
            probabilities=probs,
            thresholds=self.thresholds,
            provenance=self.provenance,
            ood_decision=ood,
            evidence=evidence,
            require_evidence_for_positive=True,
        )
        if (
            self.vlm_engine is None
            or not ood.accepted
            or ood.warning_only
            or rule_based.abstained
        ):
            return rule_based
        response, _ = answer_with_optional_vlm(
            image=_ensure_pil(image),
            question=question,
            probabilities=probs,
            thresholds=self.thresholds,
            provenance=self.provenance,
            evidence=evidence,
            vlm_engine=self.vlm_engine,
        )
        return response

    def _probabilities(self, image: Image.Image | np.ndarray) -> np.ndarray:
        if self.fixed_probabilities is not None:
            return np.asarray(self.fixed_probabilities, dtype=np.float64)
        arr = np.asarray(image.convert("L") if isinstance(image, Image.Image) else image)
        digest = hashlib.sha256(arr.tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        probs = rng.uniform(0.43, 0.47, size=len(NIH_LABELS))
        return probs.astype(np.float64)

    def _maybe_load_vlm(self, enable_vlm: bool | None) -> VLMInferenceEngine | None:
        requested = bool(self.vlm_config.get("vlm", {}).get("enabled", False))
        if enable_vlm is not None:
            requested = enable_vlm
        if not requested:
            return None
        try:
            return load_vlm(self.vlm_config)
        except Exception:
            return None


def create_app(
    service: MedGuardService | None = None,
    include_debug_routes: bool = False,
) -> FastAPI:
    """Create the Phase 4 FastAPI app."""

    resolved_service = service or MedGuardService()
    app = FastAPI(title="MedGuard-CXR", version="0.4.0")
    app.state.service = resolved_service

    @app.get("/health")
    def health() -> dict[str, Any]:
        return resolved_service.health().model_dump()

    @app.post("/predict", response_model=None)
    def predict(request: ImageRequest) -> dict[str, Any]:
        image = _decode_image_request(request)
        return resolved_service.predict(image).model_dump()

    @app.post("/explain", response_model=None)
    def explain(request: ExplainRequest) -> dict[str, Any] | JSONResponse:
        image = _decode_image_request(request)
        response = resolved_service.explain(image, request.class_name)
        if (
            response.evidence is None
            and request.class_name in resolved_service.last_predictions
            and resolved_service.last_predictions[request.class_name].prediction == 1
        ):
            return JSONResponse(
                _problem_payload(
                    409,
                    "Evidence unavailable for the previous positive prediction.",
                    resolved_service.provenance,
                ),
                status_code=409,
            )
        return response.model_dump()

    @app.post("/vqa", response_model=None)
    def vqa(request: VQARequest) -> dict[str, Any]:
        image = _decode_image_request(request)
        return resolved_service.vqa(image, request.question).model_dump()

    if include_debug_routes:

        @app.get("/debug/missing-disclaimer")
        def missing_disclaimer() -> dict[str, str]:
            return {"status": "unsafe"}

        @app.get("/debug/missing-provenance")
        def missing_provenance() -> dict[str, str]:
            return {"status": "ok", "safety_disclaimer": SAFETY_DISCLAIMER}

    app.add_middleware(AuditLogMiddleware, log_path="logs/api_audit.jsonl")
    app.add_middleware(ProvenanceMiddleware, provenance=resolved_service.provenance)
    app.add_middleware(DisclaimerMiddleware, provenance=resolved_service.provenance)
    return app


def default_model_provenance() -> ModelProvenance:
    """Return conservative smoke provenance for Phase 4A."""

    return ModelProvenance(
        classifier_checkpoint_sha256=_sha256_if_exists(Path("checkpoints/baseline_nih_best.pt")),
        calibrator_sha256=_sha256_if_exists(Path("calibrators/nih_temp_scaling.pkl")),
        is_smoke=True,
        warning=SMOKE_WARNING,
    )


def _decode_image_request(request: ImageRequest) -> Image.Image:
    if request.image_path:
        return Image.open(request.image_path).convert("RGB")
    if request.image_url:
        raise HTTPException(status_code=400, detail="image_url is disabled for local Phase 4 demo.")
    if not request.image:
        raise HTTPException(
            status_code=400,
            detail="An image, image_path, or image_url is required.",
        )
    payload = request.image
    if "," in payload and payload.split(",", maxsplit=1)[0].startswith("data:"):
        payload = payload.split(",", maxsplit=1)[1]
    try:
        raw = base64.b64decode(payload)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode image payload.") from exc


def _ensure_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")


def _ood_payload(decision: OODDecision) -> OODPayload:
    return OODPayload(
        accepted=decision.accepted,
        reason=decision.reason,  # type: ignore[arg-type]
        score=decision.score,
        warning_only=decision.warning_only,
    )


def _smoke_evidence(class_name: str | None) -> EvidencePayload | None:
    if class_name not in NIH_LABELS:
        return None
    image = Image.new("RGB", (16, 16), color=(32, 32, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return EvidencePayload(
        class_name=class_name,
        cam_uri=uri,
        bbox_normalized=(0.25, 0.25, 0.75, 0.75),
        cam_method="gradcam",
    )


def _class_from_question(question: str) -> str | None:
    lowered = question.replace("_", " ").lower()
    for class_name in NIH_LABELS:
        if class_name.replace("_", " ").lower() in lowered:
            return class_name
    return None


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text()) or {}


def _sha256_if_exists(path: Path) -> str:
    if not path.exists():
        return "unavailable"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _problem_payload(status: int, detail: str, provenance: ModelProvenance) -> dict[str, Any]:
    return ProblemDetails(
        title="MedGuard-CXR request rejected",
        status=status,
        detail=detail,
        reason="evidence_unavailable",
        model_provenance=provenance,
    ).model_dump()


app = create_app()
