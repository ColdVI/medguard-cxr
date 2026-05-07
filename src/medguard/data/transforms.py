"""Image transform factories for chest X-ray inputs.

Phase 1 implements resize, optional train-time horizontal flip, channel handling,
and normalization from YAML configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from monai.transforms import Compose, RandFlip, Resize
except ImportError:  # pragma: no cover - dependencies are installed in normal use.
    Compose = None
    RandFlip = None
    Resize = None

PHASE = "1"


def is_available() -> bool:
    """Return whether transform construction is implemented."""
    return True


class CXRImageTransform:
    """Callable transform for grayscale CXR images.

    Inputs may be a path, PIL image, NumPy array, or tensor. Outputs are
    ``torch.float32`` tensors in ``[C, H, W]`` form.
    """

    def __init__(self, config: Mapping[str, Any], train: bool) -> None:
        preprocessing = config.get("preprocessing", {})
        normalization = preprocessing.get("normalization", {})
        augmentations = preprocessing.get("train_augmentations", {})

        self.image_size = int(preprocessing.get("image_size", 224))
        self.channels = int(preprocessing.get("channels", 3))
        self.normalization = str(normalization.get("active", "imagenet")).lower()
        self.imagenet_mean = tuple(float(v) for v in normalization.get("imagenet_mean", []))
        self.imagenet_std = tuple(float(v) for v in normalization.get("imagenet_std", []))
        self.train = train
        self.flip_probability = (
            float(augmentations.get("random_horizontal_flip_probability", 0.0))
            if augmentations.get("random_horizontal_flip", False)
            else 0.0
        )

        self._monai_transform = self._build_monai_transform()

    def __call__(self, image: str | Path | Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Transform one image to a normalized tensor."""
        tensor = self._to_grayscale_tensor(image)
        tensor = self._resize_and_maybe_flip(tensor)
        tensor = self._set_channels(tensor)
        tensor = self._normalize(tensor)
        return tensor.to(dtype=torch.float32).contiguous()

    def _build_monai_transform(self) -> Any | None:
        if Compose is None or Resize is None or RandFlip is None:
            return None

        steps: list[Any] = [Resize(spatial_size=(self.image_size, self.image_size))]
        if self.train and self.flip_probability > 0:
            steps.append(RandFlip(prob=self.flip_probability, spatial_axis=1))
        return Compose(steps)

    def _resize_and_maybe_flip(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._monai_transform is not None:
            transformed = self._monai_transform(tensor)
            return torch.as_tensor(transformed, dtype=torch.float32)

        resized = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if self.train and self.flip_probability > 0 and torch.rand(()) < self.flip_probability:
            resized = torch.flip(resized, dims=(-1,))
        return resized

    def _set_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.channels == tensor.shape[0]:
            return tensor
        if self.channels == 1:
            return tensor[:1]
        if tensor.shape[0] == 1:
            return tensor.repeat(self.channels, 1, 1)
        raise ValueError(f"Cannot convert {tensor.shape[0]} channels to {self.channels} channels.")

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalization == "imagenet":
            if len(self.imagenet_mean) != self.channels or len(self.imagenet_std) != self.channels:
                raise ValueError("ImageNet normalization requires one mean/std value per channel.")
            mean = torch.tensor(self.imagenet_mean, dtype=tensor.dtype).view(self.channels, 1, 1)
            std = torch.tensor(self.imagenet_std, dtype=tensor.dtype).view(self.channels, 1, 1)
            return (tensor - mean) / std.clamp_min(1e-12)

        if self.normalization == "torchxrayvision":
            # TorchXRayVision models conventionally consume images scaled to roughly [-1024, 1024].
            return tensor * 2048.0 - 1024.0

        if self.normalization in {"none", "identity"}:
            return tensor

        raise ValueError(f"Unsupported normalization mode: {self.normalization}")

    @staticmethod
    def _to_grayscale_tensor(
        image: str | Path | Image.Image | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(image, str | Path):
            with Image.open(image) as loaded:
                return CXRImageTransform._pil_to_tensor(loaded)

        if isinstance(image, Image.Image):
            return CXRImageTransform._pil_to_tensor(image)

        if isinstance(image, np.ndarray):
            array = image.astype(np.float32, copy=False)
            if array.ndim == 3:
                array = array.mean(axis=-1)
            max_value = (
                float(np.iinfo(image.dtype).max)
                if np.issubdtype(image.dtype, np.integer)
                else 1.0
            )
            if max_value > 1:
                array = array / max_value
            return torch.from_numpy(array).unsqueeze(0).clamp(0.0, 1.0)

        tensor = image.detach().clone().to(dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected image tensor with 2 or 3 dimensions, got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)


def build_image_transform(config: Mapping[str, Any], train: bool) -> CXRImageTransform:
    """Build the configured image transform."""
    return CXRImageTransform(config=config, train=train)
