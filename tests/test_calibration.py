"""Phase 2 calibrator tests."""

from __future__ import annotations

import numpy as np

from medguard.eval.calibration_metrics import expected_calibration_error
from medguard.models.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureScalingCalibrator,
    hash_image_ids,
    load_calibrator,
    save_calibrator,
)


def test_temperature_scaling_recovers_T_eq_one_on_perfectly_calibrated_logits() -> None:
    """Soft-label calibrated logits keep temperature near one."""
    logits = np.linspace(-4.0, 4.0, 400).reshape(-1, 1)
    labels = sigmoid(logits)
    calibrator = TemperatureScalingCalibrator(classes=["A"])

    calibrator.fit(logits, labels, image_ids=[f"val-{index}" for index in range(logits.shape[0])])

    assert abs(float(calibrator.temperatures[0]) - 1.0) < 0.05


def test_temperature_scaling_reduces_ece_on_overconfident_logits() -> None:
    """Temperature scaling improves ECE on intentionally overconfident logits."""
    rng = np.random.default_rng(7)
    true_logits = rng.normal(size=(2500, 1))
    labels = rng.binomial(1, sigmoid(true_logits)).astype(float)
    overconfident_logits = true_logits * 3.0
    calibrator = TemperatureScalingCalibrator(classes=["A"])

    pre = expected_calibration_error(sigmoid(overconfident_logits), labels, n_bins=15)
    calibrator.fit(
        overconfident_logits,
        labels,
        image_ids=[f"val-{index}" for index in range(labels.shape[0])],
    )
    post = expected_calibration_error(calibrator.transform(overconfident_logits), labels, n_bins=15)

    assert post < pre
    assert calibrator.temperatures[0] > 1.0


def test_isotonic_clips_outside_training_range() -> None:
    """Isotonic calibration clips predictions beyond its fitted probability range."""
    logits = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    labels = (logits[:, 0] > 0).astype(float).reshape(-1, 1)
    calibrator = IsotonicCalibrator(classes=["A"])

    calibrator.fit(logits, labels, image_ids=[f"val-{index}" for index in range(20)])
    probs = calibrator.transform(np.array([[-20.0], [20.0]]))

    assert probs.shape == (2, 1)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert probs[0, 0] == 0.0
    assert probs[1, 0] == 1.0


def test_platt_recovers_logistic_on_logistic_data() -> None:
    """Platt scaling estimates the source logistic coefficients."""
    rng = np.random.default_rng(9)
    logits = rng.normal(size=(3000, 1))
    labels = rng.binomial(1, sigmoid(1.7 * logits - 0.4)).astype(float)
    calibrator = PlattCalibrator(classes=["A"])

    calibrator.fit(logits, labels, image_ids=[f"val-{index}" for index in range(labels.shape[0])])

    assert abs(float(calibrator.a[0]) - 1.7) < 0.25
    assert abs(float(calibrator.b[0]) + 0.4) < 0.25


def test_calibrator_save_load_roundtrip_preserves_predictions(tmp_path) -> None:
    """Pickle persistence preserves calibrated probabilities."""
    logits = np.linspace(-3.0, 3.0, 100).reshape(-1, 1)
    labels = (logits[:, 0] > 0).astype(float).reshape(-1, 1)
    calibrator = TemperatureScalingCalibrator(classes=["A"])
    calibrator.fit(logits, labels, image_ids=[f"val-{index}" for index in range(100)])
    path = tmp_path / "calibrator.pkl"

    save_calibrator(calibrator, path)
    loaded = load_calibrator(path)

    np.testing.assert_allclose(loaded.transform(logits), calibrator.transform(logits))
    assert path.with_suffix(".json").exists()


def test_calibrator_records_fit_split_hash() -> None:
    """Fit metadata records the validation split hash for leakage audits."""
    logits = np.array([[-1.0], [1.0], [2.0], [-2.0]])
    labels = np.array([[0.0], [1.0], [1.0], [0.0]])
    image_ids = ["b", "a", "d", "c"]
    calibrator = TemperatureScalingCalibrator(classes=["A"])

    calibrator.fit(logits, labels, image_ids=image_ids)

    assert calibrator.n_fit_samples == 4
    assert calibrator.fit_split_hash == hash_image_ids(image_ids)


def test_degenerate_class_does_not_error() -> None:
    """All-zero or all-one class columns are skipped, not fatal."""
    logits = np.array(
        [
            [-1.0, -2.0],
            [1.0, -1.0],
            [2.0, 0.5],
            [-2.0, 1.5],
        ]
    )
    labels = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )
    calibrator = TemperatureScalingCalibrator(classes=["A", "B"])

    calibrator.fit(logits, labels, image_ids=["a", "b", "c", "d"])

    assert calibrator.degenerate_classes == ["B"]
    assert calibrator.temperatures[1] == 1.0
    assert calibrator.transform(logits).shape == labels.shape


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Stable-enough sigmoid for test fixtures."""
    return 1.0 / (1.0 + np.exp(-values))
