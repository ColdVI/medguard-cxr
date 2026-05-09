"""Phase 4 FastAPI contract tests."""

import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from medguard.api.app import MedGuardService, create_app
from medguard.api.schemas import SAFETY_DISCLAIMER, ExplainResponse
from medguard.data.nih import NIH_LABELS


def _image_payload(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _cxr_image() -> Image.Image:
    y, x = np.indices((96, 96))
    lung = np.exp(-(((x - 48) ** 2) / 900 + ((y - 50) ** 2) / 1400))
    arr = ((0.25 + 0.5 * lung) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _natural_image() -> Image.Image:
    arr = np.zeros((96, 96, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, 96).astype(np.uint8).reshape(1, -1)
    arr[..., 1] = 20
    arr[..., 2] = 240
    return Image.fromarray(arr, mode="RGB")


def _client(probability: float = 0.1) -> TestClient:
    probs = np.full(len(NIH_LABELS), 0.1, dtype=np.float64)
    probs[NIH_LABELS.index("Pneumothorax")] = probability
    return TestClient(create_app(service=MedGuardService(fixed_probabilities=probs)))


def test_health_endpoint_returns_disclaimer_and_provenance() -> None:
    response = _client().get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["safety_disclaimer"] == SAFETY_DISCLAIMER
    assert payload["model_provenance"]["is_smoke"] is True


def test_predict_endpoint_returns_14_class_results() -> None:
    response = _client().post("/predict", json={"image": _image_payload(_cxr_image())})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 14


def test_default_smoke_predict_abstains_for_every_class() -> None:
    client = TestClient(create_app(service=MedGuardService()))

    response = client.post("/predict", json={"image": _image_payload(_cxr_image())})

    assert response.status_code == 200
    assert all(prediction["abstained"] is True for prediction in response.json()["predictions"])


def test_predict_endpoint_rejects_natural_image_with_ood_reason() -> None:
    response = _client().post("/predict", json={"image": _image_payload(_natural_image())})

    assert response.status_code == 200
    assert response.json()["predictions"] is None
    assert response.json()["ood"]["reason"] == "ood_natural_image"


def test_explain_endpoint_returns_null_for_abstained_class() -> None:
    client = _client(probability=0.5)

    response = client.post(
        "/explain",
        json={"image": _image_payload(_cxr_image()), "class_name": "Pneumothorax"},
    )

    assert response.status_code == 200
    assert response.json()["evidence"] is None
    assert response.json()["reason"] == "evidence_unavailable"


def test_explain_endpoint_refuses_inconsistent_followup() -> None:
    class BrokenEvidenceService(MedGuardService):
        def explain(self, image, class_name):  # noqa: ANN001
            return ExplainResponse(
                evidence=None,
                reason="evidence_unavailable",
                model_provenance=self.provenance,
                safety_disclaimer=SAFETY_DISCLAIMER,
            )

    probs = np.full(len(NIH_LABELS), 0.1, dtype=np.float64)
    probs[NIH_LABELS.index("Pneumothorax")] = 0.82
    service = BrokenEvidenceService(fixed_probabilities=probs)
    client = TestClient(create_app(service=service))
    image = _image_payload(_cxr_image())

    assert client.post("/predict", json={"image": image}).status_code == 200
    response = client.post(
        "/explain",
        json={"image": image, "class_name": "Pneumothorax"},
    )

    assert response.status_code == 409
    assert response.json()["safety_disclaimer"] == SAFETY_DISCLAIMER


def test_vqa_endpoint_returns_full_schema() -> None:
    response = _client(probability=0.82).post(
        "/vqa",
        json={
            "image": _image_payload(_cxr_image()),
            "question": "Is there evidence of Pneumothorax?",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert set(payload) >= {
        "question",
        "answer",
        "confidence",
        "evidence",
        "abstained",
        "reason",
        "safety_disclaimer",
        "model_provenance",
    }


def test_disclaimer_middleware_replaces_missing_disclaimer_with_500() -> None:
    client = TestClient(create_app(include_debug_routes=True))

    response = client.get("/debug/missing-disclaimer")

    assert response.status_code == 500
    assert response.json()["safety_disclaimer"] == SAFETY_DISCLAIMER


def test_provenance_middleware_injects_model_provenance() -> None:
    client = TestClient(create_app(include_debug_routes=True))

    response = client.get("/debug/missing-provenance")

    assert response.status_code == 200
    assert response.json()["model_provenance"]["warning"] == (
        "synthetic_smoke_only_not_a_real_evaluation"
    )
