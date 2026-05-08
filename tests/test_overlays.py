"""Phase 3 tests for overlay rendering."""

from pathlib import Path

import numpy as np
from PIL import Image
from scripts.generate_synthetic_overlays import main as generate_synthetic_overlays

from medguard.explain.overlays import add_banner, overlay_heatmap, save_overlay, save_overlay_grid


def test_overlay_heatmap_returns_rgb_image() -> None:
    """Overlay rendering returns an RGB image with original dimensions."""

    image = np.ones((16, 20), dtype=np.float32) * 0.5
    heatmap = np.zeros((16, 20), dtype=np.float32)
    heatmap[4:8, 6:10] = 1.0

    rendered = overlay_heatmap(image, heatmap)

    assert isinstance(rendered, Image.Image)
    assert rendered.mode == "RGB"
    assert rendered.size == (20, 16)


def test_save_overlay_writes_png_with_optional_boxes(tmp_path: Path) -> None:
    """Predicted and ground-truth boxes can be rendered to disk."""

    image = np.ones((16, 20), dtype=np.float32) * 0.5
    heatmap = np.zeros((16, 20), dtype=np.float32)
    heatmap[4:8, 6:10] = 1.0
    output_path = tmp_path / "overlay.png"

    saved = save_overlay(
        image=image,
        heatmap=heatmap,
        output_path=output_path,
        predicted_box=(0.2, 0.2, 0.5, 0.6),
        ground_truth_box=(0.25, 0.25, 0.55, 0.65),
        banner_text="SYNTHETIC SMOKE - NOT A CLINICAL EVALUATION",
    )

    assert saved == output_path
    assert output_path.exists()
    assert Image.open(output_path).size == (20, 38)


def test_save_overlay_grid_writes_4_by_4_png(tmp_path: Path) -> None:
    """The 4x4 grid helper writes an inspectable sample sheet."""

    samples = [Image.new("RGB", (8, 6), color=(index, index, index)) for index in range(16)]
    output_path = tmp_path / "grid.png"

    saved = save_overlay_grid(samples, output_path, columns=4)

    assert saved == output_path
    assert Image.open(output_path).size == (32, 24)


def test_add_banner_extends_image_height() -> None:
    """Synthetic markers are visible as an added top banner."""

    image = Image.new("RGB", (20, 16), color=(0, 0, 0))
    rendered = add_banner(image, "SYNTHETIC SMOKE")

    assert rendered.size == (20, 38)


def test_generate_synthetic_overlays_writes_audit_corpus(tmp_path: Path) -> None:
    """The Phase 3 smoke generator writes at least 20 samples and a grid."""

    generate_synthetic_overlays(
        [
            "--output-dir",
            str(tmp_path),
            "--count",
            "20",
            "--seed",
            "2026",
        ]
    )

    assert len(list(tmp_path.glob("sample_*.png"))) == 20
    assert (tmp_path / "grid.png").exists()
