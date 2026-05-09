"""Overlay rendering utilities for Phase 3 localization review artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PHASE = "3"


def is_available() -> bool:
    """Return True once Phase 3 overlay helpers are implemented."""

    return True


def overlay_heatmap(
    image: str | Path | Image.Image | np.ndarray | torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.35,
    colormap: str = "jet",
) -> Image.Image:
    """Overlay a heatmap on an X-ray image and return an RGB PIL image."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    base = _to_rgb_image(image)
    heatmap_image = colorize_heatmap(heatmap, size=base.size, colormap=colormap)
    return Image.blend(base, heatmap_image, alpha=alpha)


def colorize_heatmap(
    heatmap: np.ndarray,
    size: tuple[int, int] | None = None,
    colormap: str = "jet",
) -> Image.Image:
    """Map a normalized heatmap to RGB colors."""

    arr = _normalize_heatmap(heatmap)
    try:
        import matplotlib.colormaps as colormaps

        cmap = colormaps.get_cmap(colormap)
        rgba = cmap(arr)
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    except Exception:
        red = (arr * 255.0).astype(np.uint8)
        blue = ((1.0 - arr) * 255.0).astype(np.uint8)
        green = np.zeros_like(red)
        rgb = np.stack([red, green, blue], axis=-1)
    image = Image.fromarray(rgb, mode="RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def resize_heatmap_to_image(heatmap: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize a 2D CAM to PIL image size ``(width, height)``."""

    if len(image_size) != 2:
        raise ValueError("image_size must be (width, height).")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size values must be positive.")
    image = Image.fromarray((_normalize_heatmap(heatmap) * 255.0).astype(np.uint8), mode="L")
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def denormalize_bbox(
    bbox: Sequence[float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Convert normalized ``xyxy`` coordinates to drawable pixel endpoints."""

    return _box_to_pixels(bbox, image_size, normalized=True)


def draw_bounding_box(
    image: Image.Image,
    bbox: Sequence[float],
    color: str | tuple[int, int, int],
    width: int = 3,
    label: str | None = None,
    normalized: bool = True,
) -> Image.Image:
    """Draw one bounding box on an image and return the same image."""

    if width < 1:
        raise ValueError("width must be >= 1.")
    draw = ImageDraw.Draw(image)
    xyxy = denormalize_bbox(bbox, image.size) if normalized else _box_to_pixels(
        bbox,
        image.size,
        normalized=False,
    )
    draw.rectangle(xyxy, outline=color, width=width)
    if label:
        draw.text((xyxy[0] + 2, max(0, xyxy[1] - 12)), label, fill=color)
    return image


def add_banner(
    image: Image.Image,
    text: str,
    background: tuple[int, int, int] = (185, 28, 28),
    fill: tuple[int, int, int] = (255, 255, 255),
    height: int = 22,
) -> Image.Image:
    """Add a top banner to mark synthetic or non-clinical review artifacts."""

    if height < 1:
        raise ValueError("height must be >= 1.")
    canvas = Image.new("RGB", (image.width, image.height + height), color=background)
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), text, fill=fill)
    canvas.paste(image.convert("RGB"), (0, height))
    return canvas


def save_overlay(
    image: str | Path | Image.Image | np.ndarray | torch.Tensor,
    heatmap: np.ndarray,
    output_path: str | Path,
    predicted_box: Sequence[float] | None = None,
    ground_truth_box: Sequence[float] | Sequence[Sequence[float]] | None = None,
    alpha: float = 0.35,
    colormap: str = "jet",
    banner_text: str | None = None,
) -> Path:
    """Save a heatmap overlay with optional predicted and ground-truth boxes.

    Predicted boxes are red; ground-truth boxes are green. The visual is a
    review artifact only and intentionally avoids clinical claims in-band.
    """

    rendered = overlay_heatmap(image=image, heatmap=heatmap, alpha=alpha, colormap=colormap)
    if predicted_box is not None:
        draw_bounding_box(rendered, predicted_box, color=(220, 38, 38), label="pred")
    for index, box in enumerate(_as_box_list(ground_truth_box)):
        draw_bounding_box(
            rendered,
            box,
            color=(22, 163, 74),
            label="gt" if index == 0 else None,
        )
    if banner_text:
        rendered = add_banner(rendered, banner_text)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(path)
    return path


def save_overlay_grid(
    samples: Sequence[Image.Image],
    output_path: str | Path,
    columns: int = 4,
    background: tuple[int, int, int] = (0, 0, 0),
    banner_text: str | None = None,
) -> Path:
    """Save a grid of already-rendered overlay images."""

    if not samples:
        raise ValueError("samples must contain at least one image.")
    if columns < 1:
        raise ValueError("columns must be >= 1.")
    widths = {image.width for image in samples}
    heights = {image.height for image in samples}
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError("all samples must share the same dimensions.")

    tile_width = samples[0].width
    tile_height = samples[0].height
    rows = int(np.ceil(len(samples) / columns))
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), color=background)
    for index, sample in enumerate(samples):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(sample.convert("RGB"), (x, y))
    if banner_text:
        canvas = add_banner(canvas, banner_text)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def _to_rgb_image(image: str | Path | Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
    if isinstance(image, str | Path):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().float().numpy()
        if arr.ndim == 3:
            arr = np.moveaxis(arr, 0, -1)
        return _array_to_rgb(arr)
    return _array_to_rgb(np.asarray(image))


def _array_to_rgb(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        normalized = _normalize_to_uint8(arr)
        return Image.fromarray(normalized, mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[-1] in {1, 3}:
        normalized = _normalize_to_uint8(arr)
        if normalized.shape[-1] == 1:
            normalized = normalized[..., 0]
            return Image.fromarray(normalized, mode="L").convert("RGB")
        return Image.fromarray(normalized, mode="RGB")
    raise ValueError("image array must be 2D or HWC with 1 or 3 channels.")


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(values).all():
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    if values.max(initial=0.0) <= 1.0 and values.min(initial=0.0) >= 0.0:
        return np.clip(values * 255.0, 0, 255).astype(np.uint8)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return np.zeros(values.shape, dtype=np.uint8)
    return ((values - min_value) / (max_value - min_value) * 255.0).astype(np.uint8)


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("heatmap must be 2D.")
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    min_value = float(arr.min())
    max_value = float(arr.max())
    if max_value <= min_value:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - min_value) / (max_value - min_value), 0.0, 1.0)


def _box_to_pixels(
    bbox: Sequence[float],
    size: tuple[int, int],
    normalized: bool,
) -> tuple[int, int, int, int]:
    arr = np.asarray(bbox, dtype=np.float64)
    if arr.shape != (4,) or not np.all(np.isfinite(arr)):
        raise ValueError("bbox must contain four finite xyxy values.")
    width, height = size
    if normalized:
        x_min, y_min, x_max, y_max = arr
        arr = np.array(
            [
                np.floor(x_min * width),
                np.floor(y_min * height),
                np.ceil(x_max * width) - 1.0,
                np.ceil(y_max * height) - 1.0,
            ]
        )
    x_min, y_min, x_max, y_max = arr
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bbox must satisfy x_max > x_min and y_max > y_min.")
    return (
        int(np.clip(round(x_min), 0, width - 1)),
        int(np.clip(round(y_min), 0, height - 1)),
        int(np.clip(round(x_max), 0, width - 1)),
        int(np.clip(round(y_max), 0, height - 1)),
    )


def _as_box_list(
    boxes: Sequence[float] | Sequence[Sequence[float]] | None,
) -> list[Sequence[float]]:
    if boxes is None:
        return []
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.shape == (4,):
        return [arr.tolist()]
    if arr.ndim == 2 and arr.shape[1] == 4:
        return [row.tolist() for row in arr]
    raise ValueError("ground_truth_box must be one xyxy box or a sequence of xyxy boxes.")
