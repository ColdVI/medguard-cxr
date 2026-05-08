"""Phase 2 abstention policy tests."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from medguard.safety.abstention import (
    AbstentionThresholds,
    PredictionRecord,
    apply_abstention,
    load_thresholds_from_config,
    policy_selective_risk_curve,
    selective_risk_curve,
)


def test_low_confidence_in_band_abstains() -> None:
    """Probabilities inside the abstention band are refused."""
    thresholds = thresholds_for(["A"], [0.3], [0.7])

    record = apply_abstention(np.array([[0.5]]), thresholds)[0][0]

    assert record.prediction is None
    assert record.abstained is True
    assert record.reason == "low_confidence_band"


def test_high_confidence_above_band_does_not_abstain() -> None:
    """Positive probabilities above tau_hi become positive predictions."""
    thresholds = thresholds_for(["A"], [0.3], [0.7])

    record = apply_abstention(np.array([[0.9]]), thresholds)[0][0]

    assert record.prediction == 1
    assert record.abstained is False
    assert record.reason == ""


def test_low_confidence_below_band_does_not_abstain() -> None:
    """Probabilities below tau_lo become confident negative predictions."""
    thresholds = thresholds_for(["A"], [0.3], [0.7])

    record = apply_abstention(np.array([[0.1]]), thresholds)[0][0]

    assert record.prediction == 0
    assert record.abstained is False
    assert record.reason == ""


def test_per_class_thresholds_are_independent() -> None:
    """The same probability can abstain for one class and not another."""
    thresholds = thresholds_for(["A", "B"], [0.3, 0.2], [0.7, 0.35], rare=["B"])

    records = apply_abstention(np.array([[0.4, 0.4]]), thresholds)[0]

    assert records[0].abstained is True
    assert records[1].abstained is False
    assert records[1].prediction == 1


def test_global_threshold_is_rejected() -> None:
    """Scalar thresholds are invalid; only per-class arrays are accepted."""
    with pytest.raises(ValueError, match="per-class array"):
        AbstentionThresholds(
            classes=["A", "B"],
            tau_lo=np.array(0.3),
            tau_hi=np.array([0.7, 0.7]),
            rare_classes=[],
        )


def test_rare_class_asymmetry_enforced() -> None:
    """Rare-class positive thresholds must be lower than the default upper threshold."""
    config = {
        "classes": ["A", "Rare"],
        "abstention": {
            "default": {"tau_lo": 0.3, "tau_hi": 0.7},
            "rare_classes": {"Rare": {"tau_lo": 0.2, "tau_hi": 0.7}},
        },
    }

    with pytest.raises(ValueError, match="Rare-class tau_hi"):
        load_thresholds_from_config(config)


def test_selective_risk_is_monotone_non_increasing() -> None:
    """Risk does not rise as the retained set narrows to higher-confidence samples."""
    probs = np.array(
        [
            [0.99, 0.01],
            [0.95, 0.05],
            [0.85, 0.15],
            [0.75, 0.25],
            [0.60, 0.40],
        ]
    )
    labels = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    curve = selective_risk_curve(probs, labels, n_points=6, class_names=["A", "B"])

    assert set(curve["per_class"]) == {"A", "B"}
    for series in [curve["macro"], *curve["per_class"].values()]:
        assert "raw_risk" in series
        assert "envelope_risk" in series
        risks = [risk for risk in series["risk"] if risk is not None]
        assert all(right <= left + 1e-6 for left, right in zip(risks, risks[1:], strict=False))


def test_policy_selective_risk_uses_configured_threshold_bands() -> None:
    """Policy curve reflects per-class tau bands rather than only distance from 0.5."""
    probs = np.array([[0.4, 0.4], [0.8, 0.4]])
    labels = np.array([[1.0, 1.0], [1.0, 1.0]])
    thresholds = thresholds_for(["A", "Rare"], [0.3, 0.2], [0.7, 0.35], rare=["Rare"])

    curve = policy_selective_risk_curve(probs, labels, thresholds, n_points=5)

    assert curve["scope"] == "configured_per_class_tau_band"
    assert curve["per_class"]["A"]["coverage"][0] == 1.0
    assert curve["per_class"]["A"]["coverage"][2] == 0.5
    assert curve["per_class"]["A"]["coverage"][4] == 0.5
    assert curve["per_class"]["Rare"]["coverage"][0] == 1.0
    assert curve["per_class"]["Rare"]["coverage"][2] == 0.0
    assert curve["per_class"]["Rare"]["coverage"][4] == 1.0
    assert "raw_risk" in curve["per_class"]["Rare"]
    assert "envelope_risk" in curve["per_class"]["Rare"]


def test_abstention_fields_match_master_prompt_format() -> None:
    """PredictionRecord exposes the exact Phase 2 decision fields."""
    assert [field.name for field in fields(PredictionRecord)] == [
        "image_id",
        "class_name",
        "prediction",
        "confidence",
        "abstained",
        "reason",
    ]


def thresholds_for(
    classes: list[str],
    tau_lo: list[float],
    tau_hi: list[float],
    rare: list[str] | None = None,
) -> AbstentionThresholds:
    """Build thresholds for compact tests."""
    return AbstentionThresholds(
        classes=classes,
        tau_lo=np.asarray(tau_lo, dtype=float),
        tau_hi=np.asarray(tau_hi, dtype=float),
        rare_classes=rare or [],
    )
