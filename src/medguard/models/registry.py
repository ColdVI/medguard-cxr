"""Config-driven model registry for Academic V2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from medguard.models.backbones import build_torchvision_classifier


@dataclass(frozen=True)
class ModelSpec:
    """Stable model identity and provenance-relevant capability flags."""

    model_id: str
    family: str
    trainable: bool
    external_access: bool = False


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._builders: dict[str, Callable[[int], nn.Module]] = {}

    def register(self, spec: ModelSpec, builder: Callable[[int], nn.Module] | None) -> None:
        if spec.model_id in self._specs:
            raise ValueError(f"Model already registered: {spec.model_id}")
        self._specs[spec.model_id] = spec
        if builder is not None:
            self._builders[spec.model_id] = builder

    def build(self, model_id: str, num_classes: int) -> nn.Module:
        if model_id not in self._specs:
            raise KeyError(f"Unknown model {model_id!r}")
        if model_id not in self._builders:
            raise RuntimeError(
                f"Model {model_id!r} requires external access and has no verified local backend"
            )
        return self._builders[model_id](num_classes)

    def spec(self, model_id: str) -> ModelSpec:
        return self._specs[model_id]

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)


MODELS = ModelRegistry()


def register_builtin_models() -> ModelRegistry:
    if MODELS.names():
        return MODELS
    for model_id, family in (
        ("densenet121_v1", "cnn_legacy"),
        ("densenet121_modern", "cnn"),
        ("efficientnet_v2_s", "cnn"),
        ("convnext_tiny", "cnn"),
        ("swin_tiny", "vision_transformer"),
    ):
        architecture = "densenet121" if model_id.startswith("densenet121") else model_id
        MODELS.register(
            ModelSpec(model_id=model_id, family=family, trainable=True),
            lambda classes, architecture=architecture: build_torchvision_classifier(
                architecture, classes
            ),
        )
    MODELS.register(
        ModelSpec("eva_x_s", "cxr_self_supervised_vit", True, external_access=True),
        None,
    )
    MODELS.register(
        ModelSpec("cxr_foundation", "cxr_foundation_embedding", False, external_access=True),
        None,
    )
    return MODELS
