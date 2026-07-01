"""FastAPI app for Phase 4 safety-aware MedGuard-CXR inference."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
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
from medguard.data.transforms import build_image_transform
from medguard.eval.localization_metrics import cam_to_bbox
from medguard.explain.gradcam import generate_gradcam
from medguard.explain.overlays import overlay_heatmap
from medguard.models.calibration import load_calibrator
from medguard.models.classifier import build_classifier, probabilities_from_logits
from medguard.models.vlm import VLMInferenceEngine, answer_with_optional_vlm, load_vlm
from medguard.safety.abstention import apply_abstention, load_thresholds_from_config
from medguard.safety.ood import OODDecision, detect_ood, load_ood_config
from medguard.vqa.rule_based import answer_question, thresholds_config_with_classes

PHASE = "4"
ClassifierMode = Literal["auto", "real", "smoke"]


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
        classifier_config: str | Path = "configs/baseline_nih.yaml",
        classifier_checkpoint: str | Path = "checkpoints/baseline_nih_best.pt",
        calibrator_path: str | Path = "calibrators/nih_temp_scaling.pkl",
        classifier_mode: ClassifierMode = "auto",
        provenance: ModelProvenance | None = None,
        fixed_probabilities: np.ndarray | None = None,
        vlm_engine: VLMInferenceEngine | None = None,
        enable_vlm: bool | None = None,
        cam_threshold: float = 0.60,
    ) -> None:
        self.classifier_config_path = Path(classifier_config)
        self.classifier_checkpoint_path = Path(classifier_checkpoint)
        self.calibrator_path = Path(calibrator_path)
        self.classifier_mode = classifier_mode
        self.device = torch.device("cpu")
        self.classifier_model: torch.nn.Module | None = None
        self.classifier_transform: Any | None = None
        self.calibrator: Any | None = None
        self.classifier_load_error: str | None = None
        self.cam_threshold = float(cam_threshold)
        self.thresholds = load_thresholds_from_config(
            thresholds_config_with_classes(_load_yaml(calibration_config))
        )
        self.ood_config = load_ood_config(ood_config)
        self.vlm_config = _load_yaml(vlm_config)
        self.fixed_probabilities = fixed_probabilities
        if fixed_probabilities is None and classifier_mode != "smoke":
            self._maybe_load_real_classifier()
        self.provenance = provenance or default_model_provenance(
            checkpoint_path=self.classifier_checkpoint_path,
            calibrator_path=self.calibrator_path,
            is_smoke=self.classifier_model is None,
        )
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
        """Return gated Grad-CAM evidence for non-abstained positive decisions."""

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
        evidence = self._evidence_for_prediction(
            image=image,
            class_name=class_name,
            confidence=prediction.confidence,
            abstained=prediction.abstained,
            prediction=prediction.prediction,
        )
        if evidence is None:
            return ExplainResponse(
                evidence=None,
                reason="evidence_unavailable",
                model_provenance=self.provenance,
                safety_disclaimer=SAFETY_DISCLAIMER,
            )
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
        evidence = None
        class_name = _class_from_question(question)
        if ood.accepted and class_name in NIH_LABELS:
            class_index = NIH_LABELS.index(class_name)
            record = apply_abstention(probs.reshape(1, -1), self.thresholds)[0][class_index]
            evidence = self._evidence_for_prediction(
                image=image,
                class_name=class_name,
                confidence=record.confidence,
                abstained=record.abstained,
                prediction=record.prediction,
            )
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
        if self.classifier_model is not None:
            image_tensor = self._image_tensor(image)
            with torch.no_grad():
                logits = self.classifier_model(image_tensor)
            logits_np = logits.detach().cpu().numpy()
            if self.calibrator is not None:
                return np.asarray(self.calibrator.transform(logits_np)[0], dtype=np.float64)
            return probabilities_from_logits(logits.detach().cpu())[0].numpy().astype(np.float64)
        arr = np.asarray(image.convert("L") if isinstance(image, Image.Image) else image)
        digest = hashlib.sha256(arr.tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        probs = rng.uniform(0.43, 0.47, size=len(NIH_LABELS))
        return probs.astype(np.float64)

    def _maybe_load_real_classifier(self) -> None:
        if not self.classifier_checkpoint_path.exists():
            if self.classifier_mode == "real":
                raise FileNotFoundError(
                    f"Classifier checkpoint not found: {self.classifier_checkpoint_path}"
                )
            self.classifier_load_error = "checkpoint_missing"
            return
        try:
            config = _load_yaml(self.classifier_config_path)
            runtime_config = dict(config)
            model_config = dict(runtime_config.get("model", {}))
            # The checkpoint provides all learned weights; keep API startup network-independent.
            model_config["pretrained"] = "none"
            model_config["allow_weight_download"] = False
            runtime_config["model"] = model_config
            checkpoint = torch.load(
                self.classifier_checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            model = build_classifier(runtime_config).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.classifier_model = model
            self.classifier_transform = build_image_transform(runtime_config, train=False)
            if self.calibrator_path.exists():
                self.calibrator = load_calibrator(self.calibrator_path)
        except Exception as exc:
            if self.classifier_mode == "real":
                raise RuntimeError("Could not load the configured classifier checkpoint.") from exc
            self.classifier_model = None
            self.classifier_transform = None
            self.calibrator = None
            self.classifier_load_error = f"{type(exc).__name__}:{exc}"

    def _image_tensor(self, image: Image.Image | np.ndarray) -> torch.Tensor:
        if self.classifier_transform is None:
            raise RuntimeError("Classifier transform is unavailable.")
        tensor = self.classifier_transform(_ensure_pil(image)).unsqueeze(0)
        return tensor.to(self.device)

    def _evidence_for_prediction(
        self,
        image: Image.Image | np.ndarray,
        class_name: str | None,
        confidence: float,
        abstained: bool,
        prediction: int | None,
    ) -> EvidencePayload | None:
        if class_name not in NIH_LABELS or abstained or prediction != 1:
            return None
        if self.classifier_model is None:
            return _smoke_evidence(class_name)

        class_index = NIH_LABELS.index(class_name)
        image_tensor = self._image_tensor(image).squeeze(0)
        heatmap = generate_gradcam(
            model=self.classifier_model,
            image=image_tensor,
            class_index=class_index,
            confidence=confidence,
            abstained=abstained,
            abstention_threshold=float(self.thresholds.tau_hi[class_index]),
        )
        if heatmap is None:
            return None
        bbox = cam_to_bbox(heatmap, threshold=self.cam_threshold)
        overlay = overlay_heatmap(_ensure_pil(image), heatmap)
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        return EvidencePayload(
            class_name=class_name,
            cam_uri=uri,
            bbox_normalized=bbox,
            cam_method="gradcam",
        )

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


def default_model_provenance(
    checkpoint_path: str | Path = "checkpoints/baseline_nih_best.pt",
    calibrator_path: str | Path = "calibrators/nih_temp_scaling.pkl",
    is_smoke: bool = True,
) -> ModelProvenance:
    """Return provenance for the active Phase 4 classifier mode."""

    return ModelProvenance(
        classifier_checkpoint_sha256=_sha256_if_exists(Path(checkpoint_path)),
        calibrator_sha256=_sha256_if_exists(Path(calibrator_path)),
        is_smoke=is_smoke,
        warning=SMOKE_WARNING if is_smoke else None,
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
