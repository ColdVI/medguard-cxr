"""Phase 4 OOD gate tests."""

import socket

import numpy as np
import pytest

from medguard.safety.ood import detect_ood


def _synthetic_pa() -> np.ndarray:
    y, x = np.indices((96, 96))
    lung_left = np.exp(-(((x - 34) ** 2) / 160 + ((y - 50) ** 2) / 900))
    lung_right = np.exp(-(((x - 62) ** 2) / 160 + ((y - 50) ** 2) / 900))
    image = 0.25 + 0.45 * (lung_left + lung_right)
    return np.clip(image, 0.0, 1.0)


def test_ood_rejects_natural_image() -> None:
    image = np.zeros((96, 96, 3), dtype=np.float32)
    image[..., 0] = np.linspace(0.0, 1.0, 96).reshape(1, -1)
    image[..., 1] = 0.2
    image[..., 2] = 0.9 - image[..., 0] * 0.4

    decision = detect_ood(image, config={"natural_edge_chi2_max": 0.0})

    assert decision.accepted is False
    assert decision.reason == "ood_natural_image"


def test_ood_rejects_blank_image() -> None:
    decision = detect_ood(np.zeros((64, 64), dtype=np.float32))

    assert decision.accepted is False
    assert decision.reason == "ood_blank_image"


def test_ood_rejects_noise_image() -> None:
    rng = np.random.default_rng(2026)
    decision = detect_ood(rng.random((96, 96), dtype=np.float32))

    assert decision.accepted is False
    assert decision.reason == "ood_corrupted_image"


def test_ood_warns_on_lateral_view_when_classifier_present() -> None:
    decision = detect_ood(_synthetic_pa(), view_classifier=lambda _image: "lateral")

    assert decision.accepted is True
    assert decision.warning_only is True
    assert decision.reason == "ood_unsupported_view"


def test_ood_warns_on_lateral_view_skipped_when_classifier_absent() -> None:
    decision = detect_ood(_synthetic_pa())

    assert decision.accepted is True
    assert decision.warning_only is False
    assert decision.reason == ""


def test_ood_raises_when_view_classifier_path_has_no_callable() -> None:
    with pytest.raises(RuntimeError, match="view_classifier_path is set"):
        detect_ood(_synthetic_pa(), config={"view_classifier_path": "checkpoints/view.pt"})


def test_ood_accepts_synthetic_pa_frontal() -> None:
    decision = detect_ood(_synthetic_pa())

    assert decision.accepted is True
    assert decision.reason == ""


def test_ood_does_not_call_external_services(monkeypatch) -> None:
    def fail_network(*_args, **_kwargs):
        raise AssertionError("network must not be called")

    monkeypatch.setattr(socket, "create_connection", fail_network)

    assert detect_ood(_synthetic_pa()).accepted is True
