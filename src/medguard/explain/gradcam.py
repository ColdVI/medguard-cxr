"""Grad-CAM utilities for Phase 3 grounded localization.

These helpers expose heatmaps for engineering evaluation only. They do not
validate a diagnosis and should not be interpreted as definitive evidence.
"""

from __future__ import annotations

from contextlib import ExitStack

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PHASE = "3"


def is_available() -> bool:
    """Return True once Phase 3 Grad-CAM helpers are implemented."""

    return True


def should_generate_explanation(
    confidence: float,
    abstained: bool,
    abstention_threshold: float,
) -> bool:
    """Return whether a prediction is eligible for Grad-CAM generation."""

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1].")
    if not 0.0 <= abstention_threshold <= 1.0:
        raise ValueError("abstention_threshold must be in [0, 1].")
    return bool((not abstained) and confidence > abstention_threshold)


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the last ``nn.Conv2d`` module, before any global pooling/head."""

    last_conv: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No nn.Conv2d layer found for Grad-CAM.")
    return last_conv


def generate_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    class_index: int,
    confidence: float,
    abstained: bool,
    abstention_threshold: float,
    target_layer: nn.Module | None = None,
    method: str = "gradcam",
) -> np.ndarray | None:
    """Generate a normalized Grad-CAM heatmap for one image/class.

    Returns ``None`` when the prediction is abstained or below the configured
    confidence gate. The returned heatmap has the same height and width as the
    input image tensor.
    """

    if not should_generate_explanation(confidence, abstained, abstention_threshold):
        return None

    batch = _as_batch(image)
    if batch.shape[0] != 1:
        raise ValueError("generate_gradcam expects exactly one image.")
    if class_index < 0:
        raise ValueError("class_index must be non-negative.")

    target = target_layer or find_last_conv_layer(model)
    was_training = model.training
    model.eval()

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        activations.append(output.detach())

    def backward_hook(
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        if grad_output[0] is not None:
            gradients.append(grad_output[0].detach())

    with ExitStack() as stack:
        stack.callback(target.register_forward_hook(forward_hook).remove)
        stack.callback(target.register_full_backward_hook(backward_hook).remove)

        model.zero_grad(set_to_none=True)
        logits = model(batch)
        if logits.ndim != 2:
            raise ValueError("model output must have shape [N, C].")
        if class_index >= logits.shape[1]:
            raise ValueError("class_index is outside the model output dimension.")
        logits[:, class_index].sum().backward()

    if was_training:
        model.train()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

    cam = gradcam_from_tensors(
        activations[-1],
        gradients[-1],
        output_size=batch.shape[-2:],
        method=method,
    )
    return cam


def gradcam_from_tensors(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    output_size: tuple[int, int],
    method: str = "gradcam",
) -> np.ndarray:
    """Build a CAM heatmap from captured activation/gradient tensors."""

    normalized_method = method.lower().replace("-", "_")
    if normalized_method == "gradcam":
        return _gradcam_from_tensors(activations, gradients, output_size)
    if normalized_method in {"gradcam_plus_plus", "gradcam++"}:
        return _gradcam_plus_plus_from_tensors(activations, gradients, output_size)
    raise ValueError(f"Unsupported Grad-CAM method: {method}")


def _gradcam_from_tensors(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    output_size: tuple[int, int],
) -> np.ndarray:
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    return _normalize_and_resize_cam(cam, output_size)


def _gradcam_plus_plus_from_tensors(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    output_size: tuple[int, int],
) -> np.ndarray:
    gradients_power_2 = gradients.pow(2)
    gradients_power_3 = gradients_power_2 * gradients
    activation_sum = activations.sum(dim=(2, 3), keepdim=True)
    denominator = 2.0 * gradients_power_2 + activation_sum * gradients_power_3
    alphas = gradients_power_2 / denominator.clamp_min(1e-12)
    alphas = torch.where(torch.isfinite(alphas), alphas, torch.zeros_like(alphas))
    positive_gradients = torch.relu(gradients)
    weights = (alphas * positive_gradients).sum(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    return _normalize_and_resize_cam(cam, output_size)


def _normalize_and_resize_cam(cam: torch.Tensor, output_size: tuple[int, int]) -> np.ndarray:
    cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
    cam_2d = cam[0, 0]
    cam_min = torch.min(cam_2d)
    cam_max = torch.max(cam_2d)
    if torch.isclose(cam_max, cam_min):
        return np.zeros(output_size, dtype=np.float32)
    normalized = (cam_2d - cam_min) / (cam_max - cam_min)
    return normalized.detach().cpu().numpy().astype(np.float32)


def _as_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim == 4:
        return image
    raise ValueError("image must have shape [C, H, W] or [N, C, H, W].")
