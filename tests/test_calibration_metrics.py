"""Phase 2 calibration metric tests."""

from __future__ import annotations

import numpy as np

from medguard.eval.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    plot_reliability_diagram,
    reliability_curve,
)


def test_ece_zero_for_perfect_calibration() -> None:
    """Exact 0/1 probabilities matching labels have zero calibration error."""
    probs = np.array([0.0, 1.0, 0.0, 1.0])
    labels = np.array([0.0, 1.0, 0.0, 1.0])

    assert expected_calibration_error(probs, labels, n_bins=2) == 0.0


def test_ece_one_for_inverted_predictions() -> None:
    """Confidently inverted predictions produce the maximum ECE in two bins."""
    probs = np.array([0.0, 1.0])
    labels = np.array([1.0, 0.0])

    assert expected_calibration_error(probs, labels, n_bins=2) == 1.0


def test_mce_geq_ece() -> None:
    """MCE upper-bounds weighted-average ECE for a single class."""
    probs = np.array([0.1, 0.4, 0.7, 0.9])
    labels = np.array([0.0, 1.0, 1.0, 1.0])

    ece = expected_calibration_error(probs, labels, n_bins=4)
    mce = maximum_calibration_error(probs, labels, n_bins=4)

    assert mce >= ece


def test_brier_score_zero_when_probs_equal_labels() -> None:
    """Brier score is zero for exact probabilities."""
    probs = np.array([[0.0, 1.0], [1.0, 0.0]])
    labels = np.array([[0.0, 1.0], [1.0, 0.0]])

    np.testing.assert_allclose(brier_score(probs, labels), np.array([0.0, 0.0]))


def test_reliability_curve_returns_nan_for_empty_bins() -> None:
    """Empty reliability bins expose NaN accuracy for plotting gaps."""
    probs = np.array([0.1, 0.1])
    labels = np.array([0.0, 1.0])

    _, bin_acc, bin_count = reliability_curve(probs, labels, n_bins=3)

    assert np.isnan(bin_acc[1])
    assert np.isnan(bin_acc[2])
    assert bin_count.tolist() == [2.0, 0.0, 0.0]


def test_plot_reliability_diagram_writes_png_with_banner_when_synthetic(tmp_path) -> None:
    """Synthetic diagrams are generated as PNGs with the smoke banner path enabled."""
    rng = np.random.default_rng(11)
    probs_pre = rng.uniform(0.0, 1.0, size=(32, 14))
    probs_post = np.clip(probs_pre * 0.9 + 0.05, 0.0, 1.0)
    labels = rng.binomial(1, 0.2, size=(32, 14)).astype(float)
    path = tmp_path / "reliability.png"

    plot_reliability_diagram(
        probs_pre=probs_pre,
        probs_post=probs_post,
        labels=labels,
        class_names=[f"class_{index}" for index in range(14)],
        save_path=path,
        is_synthetic=True,
    )

    assert path.exists()
    assert path.stat().st_size > 0
