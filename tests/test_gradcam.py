"""Phase 3 tests for Grad-CAM utilities."""

import numpy as np
import torch
from torch import nn

from medguard.explain.gradcam import (
    find_last_conv_layer,
    generate_gradcam,
    gradcam_from_tensors,
    should_generate_explanation,
)


class TinyCnn(nn.Module):
    """Small CNN with a clear final convolutional target."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


def test_find_last_conv_layer_returns_final_conv() -> None:
    """Grad-CAM targets the final convolution, not the classifier head."""

    model = TinyCnn()
    assert find_last_conv_layer(model) is model.features[2]


def test_should_generate_explanation_respects_abstention_and_confidence_gate() -> None:
    """Abstained or low-confidence predictions do not get Grad-CAM."""

    assert should_generate_explanation(0.8, abstained=False, abstention_threshold=0.7) is True
    assert should_generate_explanation(0.8, abstained=True, abstention_threshold=0.7) is False
    assert should_generate_explanation(0.7, abstained=False, abstention_threshold=0.7) is False


def test_generate_gradcam_returns_input_spatial_size() -> None:
    """Generated heatmaps are normalized and match input HxW."""

    torch.manual_seed(2026)
    model = TinyCnn()
    image = torch.rand(1, 16, 20)

    heatmap = generate_gradcam(
        model,
        image=image,
        class_index=1,
        confidence=0.9,
        abstained=False,
        abstention_threshold=0.7,
    )

    assert heatmap is not None
    assert heatmap.shape == (16, 20)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_generate_gradcam_plus_plus_returns_input_spatial_size() -> None:
    """GradCAM++ is available as the secondary Phase 3 method."""

    torch.manual_seed(2026)
    model = TinyCnn()
    image = torch.rand(1, 16, 20)

    heatmap = generate_gradcam(
        model,
        image=image,
        class_index=1,
        confidence=0.9,
        abstained=False,
        abstention_threshold=0.7,
        method="gradcam_plus_plus",
    )

    assert heatmap is not None
    assert heatmap.shape == (16, 20)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_gradcam_methods_agree_on_uniform_gradient_case() -> None:
    """Uniform activations/gradients normalize to the same heatmap."""

    activations = torch.ones(1, 2, 4, 4)
    gradients = torch.ones(1, 2, 4, 4)

    vanilla = gradcam_from_tensors(activations, gradients, output_size=(8, 8), method="gradcam")
    plus_plus = gradcam_from_tensors(
        activations,
        gradients,
        output_size=(8, 8),
        method="gradcam_plus_plus",
    )

    assert np.array_equal(vanilla, plus_plus)


def test_generate_gradcam_returns_none_when_abstained() -> None:
    """The explanation path is gated before hooks run."""

    model = TinyCnn()
    image = torch.rand(1, 16, 20)

    assert (
        generate_gradcam(
            model,
            image=image,
            class_index=1,
            confidence=0.95,
            abstained=True,
            abstention_threshold=0.7,
        )
        is None
    )
