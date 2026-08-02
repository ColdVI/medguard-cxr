"""Academic V2 model registry tests."""

import pytest
import torch

from medguard.models.heads import masked_binary_cross_entropy
from medguard.models.registry import register_builtin_models


def test_model_registry_contains_required_progression() -> None:
    registry = register_builtin_models()

    assert set(registry.names()) >= {
        "densenet121_v1",
        "densenet121_modern",
        "efficientnet_v2_s",
        "convnext_tiny",
        "swin_tiny",
        "eva_x_s",
        "cxr_foundation",
    }
    with pytest.raises(RuntimeError, match="external access"):
        registry.build("cxr_foundation", 8)


def test_missing_label_mask_excludes_unsupervised_targets() -> None:
    logits = torch.tensor([[0.0, 100.0]])
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])

    masked = masked_binary_cross_entropy(logits, targets, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :1], targets[:, :1]
    )

    assert torch.allclose(masked, expected)
