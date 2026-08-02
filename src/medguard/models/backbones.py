"""Academic V2 trainable backbone builders with a uniform logits contract."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn
from torchvision.models import convnext_tiny, densenet121, efficientnet_v2_s, swin_t


def build_torchvision_classifier(
    architecture: str,
    num_classes: int,
    *,
    pretrained: bool = False,
) -> nn.Module:
    """Build supported CNN/transformer baselines returning raw ``[B,C]`` logits."""

    if pretrained:
        raise ValueError(
            "Weight downloads must be resolved by the checkpoint manager and recorded in provenance"
        )
    builders: dict[str, Callable[[], nn.Module]] = {
        "densenet121": lambda: _densenet(num_classes),
        "efficientnet_v2_s": lambda: _efficientnet(num_classes),
        "convnext_tiny": lambda: _convnext(num_classes),
        "swin_tiny": lambda: _swin(num_classes),
    }
    key = architecture.lower()
    if key not in builders:
        raise KeyError(f"Unsupported trainable backbone: {architecture}")
    return builders[key]()


def _densenet(num_classes: int) -> nn.Module:
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def _efficientnet(num_classes: int) -> nn.Module:
    model = efficientnet_v2_s(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _convnext(num_classes: int) -> nn.Module:
    model = convnext_tiny(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _swin(num_classes: int) -> nn.Module:
    model = swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
