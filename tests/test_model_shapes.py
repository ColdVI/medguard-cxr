"""Phase 1 classifier shape tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from medguard.models.classifier import build_classifier, build_loss, probabilities_from_logits


def test_classifier_forward_returns_raw_logits() -> None:
    """DenseNet121 baseline returns raw logits, not probabilities."""
    config = {
        "model": {
            "architecture": "densenet121",
            "pretrained": "none",
            "allow_weight_download": False,
            "num_classes": 14,
        }
    }
    model = build_classifier(config)
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 14)
    assert not any(isinstance(module, nn.Sigmoid | nn.Softmax) for module in model.modules())


def test_loss_is_bce_with_logits_and_sigmoid_is_explicit() -> None:
    """Loss consumes raw logits; sigmoid is reserved for evaluation/inference."""
    pos_weight = torch.ones(14)
    loss_fn = build_loss(pos_weight=pos_weight)
    logits = torch.randn(2, 14)
    labels = torch.randint(0, 2, (2, 14)).float()

    loss = loss_fn(logits, labels)
    probabilities = probabilities_from_logits(logits)

    assert isinstance(loss_fn, nn.BCEWithLogitsLoss)
    assert loss.ndim == 0
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_imagenet_without_download_warns_about_random_init() -> None:
    """Real-data configs should not silently train random DenseNet121 by accident."""
    config = {
        "model": {
            "architecture": "densenet121",
            "pretrained": "imagenet",
            "allow_weight_download": False,
            "num_classes": 14,
        }
    }

    with pytest.warns(UserWarning, match="random initialization"):
        build_classifier(config)
