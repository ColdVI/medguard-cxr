"""Generate a synthetic Phase 3 overlay corpus for visual-audit plumbing.

The generated images are synthetic smoke artifacts only. They are not clinical
evidence and must not be used to estimate localization performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from medguard.explain.overlays import save_overlay, save_overlay_grid

BANNER = "SYNTHETIC SMOKE - NOT A CLINICAL EVALUATION"
OverlaySample = tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def main(argv: list[str] | None = None) -> int:
    """Generate sample overlays and a 4x4 grid."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/overlays",
        help="Directory for sample PNGs.",
    )
    parser.add_argument("--count", type=int, default=20, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic RNG seed.")
    parser.add_argument("--grid", default="grid.png", help="Grid filename inside output-dir.")
    args = parser.parse_args(argv)

    if args.count < 20:
        raise RuntimeError("Phase 3B visual audit smoke corpus requires at least 20 overlays.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sample_paths: list[Path] = []

    for index in range(args.count):
        image, heatmap, predicted_box, ground_truth_box = _synthetic_sample(rng, index)
        sample_path = output_dir / f"sample_{index:02d}.png"
        save_overlay(
            image=image,
            heatmap=heatmap,
            output_path=sample_path,
            predicted_box=predicted_box,
            ground_truth_box=ground_truth_box,
            banner_text=BANNER,
        )
        sample_paths.append(sample_path)

    grid_samples = [Image.open(path).convert("RGB") for path in sample_paths[:16]]
    grid_path = output_dir / args.grid
    save_overlay_grid(grid_samples, grid_path, columns=4, banner_text=BANNER)
    print(f"wrote {len(sample_paths)} synthetic overlays and grid: {grid_path}")
    return 0


def _synthetic_sample(
    rng: np.random.Generator,
    index: int,
    size: tuple[int, int] = (224, 224),
) -> OverlaySample:
    height, width = size
    y, x = np.mgrid[0:height, 0:width]

    left_lung = np.exp(-(((x - width * 0.36) / 42.0) ** 2 + ((y - height * 0.52) / 72.0) ** 2))
    right_lung = np.exp(-(((x - width * 0.64) / 42.0) ** 2 + ((y - height * 0.52) / 72.0) ** 2))
    rib_texture = 0.05 * np.sin((y + index * 3) / 7.5)
    noise = rng.normal(0.0, 0.025, size=size)
    image = np.clip(0.18 + 0.55 * (left_lung + right_lung) + rib_texture + noise, 0.0, 1.0)

    cx = rng.uniform(width * 0.28, width * 0.72)
    cy = rng.uniform(height * 0.30, height * 0.72)
    sigma_x = rng.uniform(14.0, 24.0)
    sigma_y = rng.uniform(12.0, 22.0)
    heatmap = np.exp(-(((x - cx) / sigma_x) ** 2 + ((y - cy) / sigma_y) ** 2))
    heatmap += 0.15 * rng.random(size)
    heatmap = heatmap.astype(np.float32)

    gt_width = rng.uniform(0.16, 0.26)
    gt_height = rng.uniform(0.14, 0.24)
    gt_center_x = float(cx / width)
    gt_center_y = float(cy / height)
    ground_truth_box = _box_from_center(gt_center_x, gt_center_y, gt_width, gt_height)

    offset_x = rng.normal(0.0, 0.025)
    offset_y = rng.normal(0.0, 0.025)
    scale = rng.uniform(0.9, 1.15)
    predicted_box = _box_from_center(
        gt_center_x + offset_x,
        gt_center_y + offset_y,
        gt_width * scale,
        gt_height * scale,
    )
    return image.astype(np.float32), heatmap, predicted_box, ground_truth_box


def _box_from_center(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    x_min = max(0.0, center_x - width / 2.0)
    y_min = max(0.0, center_y - height / 2.0)
    x_max = min(1.0, center_x + width / 2.0)
    y_max = min(1.0, center_y + height / 2.0)
    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError("Synthetic overlay box collapsed.")
    return x_min, y_min, x_max, y_max


if __name__ == "__main__":
    raise SystemExit(main())
