"""DenseNet121 multi-label classifier for Phase 1."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

PHASE = "1"


def is_available() -> bool:
    """Return whether the Phase 1 classifier is implemented."""
    return True


class MedGuardCXRClassifier(nn.Module):
    """DenseNet121 classifier that returns raw logits.

    The model intentionally does not apply sigmoid or softmax in ``forward``.
    Use ``torch.sigmoid(logits)`` only in inference/evaluation code.
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: str | bool | None = "imagenet",
        allow_weight_download: bool = False,
    ) -> None:
        super().__init__()
        weights = _resolve_densenet_weights(pretrained, allow_weight_download)
        self.backbone = densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return raw multi-label logits with shape ``[batch, num_classes]``."""
        return self.backbone(image)


def build_classifier(config: Mapping[str, Any]) -> MedGuardCXRClassifier:
    """Build the configured classifier."""
    model_cfg = config.get("model", {})
    architecture = str(model_cfg.get("architecture", "densenet121")).lower()
    if architecture != "densenet121":
        raise ValueError(f"Unsupported Phase 1 architecture: {architecture}")
    return MedGuardCXRClassifier(
        num_classes=int(model_cfg.get("num_classes", 14)),
        pretrained=model_cfg.get("pretrained", "imagenet"),
        allow_weight_download=bool(model_cfg.get("allow_weight_download", False)),
    )


def build_loss(pos_weight: torch.Tensor | None = None) -> nn.BCEWithLogitsLoss:
    """Build the Phase 1 multi-label loss."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert raw logits to probabilities for inference/evaluation only."""
    return torch.sigmoid(logits)


def _resolve_densenet_weights(
    pretrained: str | bool | None,
    allow_weight_download: bool,
) -> DenseNet121_Weights | None:
    if pretrained in {False, None, "none", "random"}:
        return None
    if pretrained in {True, "imagenet"}:
        if allow_weight_download:
            return DenseNet121_Weights.DEFAULT
        warnings.warn(
            "DenseNet121 configured as pretrained=imagenet but allow_weight_download=false; "
            "using random initialization. This is suitable only for smoke/CI paths.",
            stacklevel=2,
        )
        return None
    raise ValueError(f"Unsupported DenseNet121 pretrained setting: {pretrained}")
