"""Per-class abstention policy for calibrated multi-label probabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

PHASE = "2"
MONOTONE_TOLERANCE = 1e-3


def is_available() -> bool:
    """Return whether abstention logic is implemented."""
    return True


@dataclass
class AbstentionThresholds:
    """Per-class lower and upper thresholds around the abstention band."""

    classes: list[str]
    tau_lo: np.ndarray
    tau_hi: np.ndarray
    rare_classes: list[str]

    def __post_init__(self) -> None:
        self.tau_lo = _validate_threshold_array(self.tau_lo, "tau_lo", len(self.classes))
        self.tau_hi = _validate_threshold_array(self.tau_hi, "tau_hi", len(self.classes))
        if np.any(self.tau_lo >= self.tau_hi):
            raise ValueError("Each class must satisfy tau_lo < tau_hi.")
        unknown_rare = set(self.rare_classes) - set(self.classes)
        if unknown_rare:
            raise ValueError(f"Rare classes not present in class list: {sorted(unknown_rare)}")


@dataclass
class PredictionRecord:
    """One per-image, per-class selective prediction record."""

    image_id: str | None
    class_name: str
    prediction: int | None
    confidence: float
    abstained: bool
    reason: str


def apply_abstention(
    probs: np.ndarray,
    thresholds: AbstentionThresholds,
    image_ids: list[str] | None = None,
) -> list[list[PredictionRecord]]:
    """Return ``N`` rows by ``C`` per-class prediction records."""
    probabilities = _validate_probs(probs, len(thresholds.classes))
    if image_ids is not None and len(image_ids) != probabilities.shape[0]:
        raise ValueError("image_ids length must match number of probability rows.")

    rows: list[list[PredictionRecord]] = []
    for row_index in range(probabilities.shape[0]):
        image_id = image_ids[row_index] if image_ids is not None else None
        records: list[PredictionRecord] = []
        for class_index, class_name in enumerate(thresholds.classes):
            confidence = float(probabilities[row_index, class_index])
            if confidence < thresholds.tau_lo[class_index]:
                prediction: int | None = 0
                abstained = False
                reason = ""
            elif confidence > thresholds.tau_hi[class_index]:
                prediction = 1
                abstained = False
                reason = ""
            else:
                prediction = None
                abstained = True
                reason = "low_confidence_band"
            records.append(
                PredictionRecord(
                    image_id=image_id,
                    class_name=class_name,
                    prediction=prediction,
                    confidence=confidence,
                    abstained=abstained,
                    reason=reason,
                )
            )
        rows.append(records)
    return rows


def selective_risk_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_points: int = 21,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute intrinsic selective risk around a symmetric 0.5 confidence band."""
    probabilities = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if probabilities.shape != targets.shape or probabilities.ndim != 2:
        raise ValueError("selective_risk_curve expects matching [N, C] arrays.")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    resolved_names = _resolve_curve_class_names(class_names, probabilities.shape[1])

    thresholds = np.linspace(0.0, 0.5, n_points)
    per_class: dict[str, dict[str, Any]] = {}
    macro_coverages: list[float] = []
    macro_risks: list[float | None] = []
    predictions = (probabilities >= 0.5).astype(np.float64)
    errors = (predictions != targets).astype(np.float64)
    confidence_distance = np.abs(probabilities - 0.5)

    for class_index, class_name in enumerate(resolved_names):
        coverage, risk = _risk_sweep(
            confidence_distance[:, class_index],
            errors[:, class_index],
            thresholds,
        )
        per_class[class_name] = _risk_payload(
            class_name=class_name,
            coverage=coverage,
            raw_risk=risk,
            sweep_values=thresholds,
            sweep_key="threshold",
        )

    for threshold in thresholds:
        retained = confidence_distance >= threshold
        macro_coverages.append(float(np.mean(retained)))
        if not np.any(retained):
            macro_risks.append(None)
        else:
            macro_risks.append(float(np.mean(errors[retained])))
    macro = _risk_payload(
        class_name="macro",
        coverage=macro_coverages,
        raw_risk=macro_risks,
        sweep_values=thresholds,
        sweep_key="threshold",
    )
    violations = _collect_curve_violations(macro, per_class)

    return {
        "scope": "intrinsic_symmetric_around_0.5",
        "macro": macro,
        "per_class": per_class,
        "monotone_risk_invariant_passed": not violations,
        "monotone_risk_invariant_violations": violations,
    }


def policy_selective_risk_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: AbstentionThresholds,
    n_points: int = 21,
) -> dict[str, Any]:
    """Compute selective risk by sweeping the configured per-class abstention bands."""
    probabilities = _validate_probs(probs, len(thresholds.classes))
    targets = np.asarray(labels, dtype=np.float64)
    if probabilities.shape != targets.shape:
        raise ValueError("policy_selective_risk_curve expects matching [N, C] arrays.")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    width_factors = np.linspace(1.0, 0.0, n_points)
    per_class: dict[str, dict[str, Any]] = {}
    macro_coverages: list[float] = []
    macro_risks: list[float | None] = []

    for class_index, class_name in enumerate(thresholds.classes):
        coverage: list[float] = []
        risk: list[float | None] = []
        for width_factor in width_factors:
            tau_lo, tau_hi = _scaled_thresholds(
                thresholds.tau_lo[class_index],
                thresholds.tau_hi[class_index],
                float(width_factor),
            )
            retained = (probabilities[:, class_index] < tau_lo) | (
                probabilities[:, class_index] > tau_hi
            )
            coverage.append(float(np.mean(retained)))
            if not np.any(retained):
                risk.append(None)
                continue
            predictions = (probabilities[:, class_index] > tau_hi).astype(np.float64)
            errors = predictions[retained] != targets[:, class_index][retained]
            risk.append(float(np.mean(errors)))
        per_class[class_name] = _risk_payload(
            class_name=class_name,
            coverage=coverage,
            raw_risk=risk,
            sweep_values=width_factors,
            sweep_key="width_factor",
        )

    for width_factor in width_factors:
        tau_lo, tau_hi = _scaled_threshold_arrays(thresholds, float(width_factor))
        retained = (probabilities < tau_lo.reshape(1, -1)) | (
            probabilities > tau_hi.reshape(1, -1)
        )
        macro_coverages.append(float(np.mean(retained)))
        if not np.any(retained):
            macro_risks.append(None)
            continue
        predictions = (probabilities > tau_hi.reshape(1, -1)).astype(np.float64)
        macro_risks.append(float(np.mean((predictions != targets)[retained])))

    macro = _risk_payload(
        class_name="macro",
        coverage=macro_coverages,
        raw_risk=macro_risks,
        sweep_values=width_factors,
        sweep_key="width_factor",
    )
    violations = _collect_curve_violations(macro, per_class)
    return {
        "scope": "configured_per_class_tau_band",
        "macro": macro,
        "per_class": per_class,
        "monotone_risk_invariant_passed": not violations,
        "monotone_risk_invariant_violations": violations,
    }


def load_thresholds_from_config(config: Mapping[str, Any]) -> AbstentionThresholds:
    """Load per-class abstention thresholds from YAML-derived config."""
    classes = _resolve_classes(config)
    abstention_cfg = config.get("abstention", {})
    default_cfg = abstention_cfg.get("default", {})
    default_lo = float(default_cfg.get("tau_lo", 0.30))
    default_hi = float(default_cfg.get("tau_hi", 0.70))
    rare_cfg = abstention_cfg.get("rare_classes", {})

    tau_lo = np.full(len(classes), default_lo, dtype=np.float64)
    tau_hi = np.full(len(classes), default_hi, dtype=np.float64)
    rare_classes: list[str] = []

    for class_name, class_cfg in rare_cfg.items():
        if class_name not in classes:
            raise ValueError(f"Rare class {class_name!r} is not in the configured class list.")
        class_tau_hi = float(class_cfg["tau_hi"])
        if class_tau_hi >= default_hi:
            raise ValueError("Rare-class tau_hi must be lower than the default tau_hi.")
        class_index = classes.index(class_name)
        tau_lo[class_index] = float(class_cfg["tau_lo"])
        tau_hi[class_index] = class_tau_hi
        rare_classes.append(class_name)

    return AbstentionThresholds(
        classes=classes,
        tau_lo=tau_lo,
        tau_hi=tau_hi,
        rare_classes=rare_classes,
    )


def _risk_sweep(
    confidence_distance: np.ndarray,
    errors: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[list[float], list[float | None]]:
    coverage: list[float] = []
    risk: list[float | None] = []
    for threshold in thresholds:
        retained = confidence_distance >= threshold
        coverage.append(float(np.mean(retained)))
        if not np.any(retained):
            risk.append(None)
        else:
            risk.append(float(np.mean(errors[retained])))
    return coverage, risk


def _monotone_non_increasing_envelope(values: list[float | None]) -> list[float | None]:
    envelope: list[float | None] = []
    best_so_far: float | None = None
    for value in values:
        if value is None:
            envelope.append(None)
            continue
        best_so_far = value if best_so_far is None else min(best_so_far, value)
        envelope.append(float(best_so_far))
    return envelope


def _risk_payload(
    class_name: str,
    coverage: list[float],
    raw_risk: list[float | None],
    sweep_values: np.ndarray,
    sweep_key: str,
) -> dict[str, Any]:
    envelope = _monotone_non_increasing_envelope(raw_risk)
    violations = _monotone_violations(class_name, raw_risk, sweep_values)
    return {
        "coverage": coverage,
        "raw_risk": raw_risk,
        "envelope_risk": envelope,
        "risk": envelope,
        sweep_key: [float(value) for value in sweep_values.tolist()],
        "monotone_risk_invariant_passed": not violations,
        "monotone_risk_invariant_violations": violations,
    }


def _monotone_violations(
    class_name: str,
    values: list[float | None],
    sweep_values: np.ndarray,
) -> list[dict[str, float | str]]:
    violations: list[dict[str, float | str]] = []
    for index, (left, right) in enumerate(zip(values, values[1:], strict=False)):
        if left is None or right is None:
            continue
        delta = right - left
        if delta > MONOTONE_TOLERANCE:
            violations.append(
                {
                    "class": class_name,
                    "threshold_left": float(sweep_values[index]),
                    "threshold_right": float(sweep_values[index + 1]),
                    "delta": float(delta),
                }
            )
    return violations


def _collect_curve_violations(
    macro: Mapping[str, Any],
    per_class: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, float | str]]:
    violations = list(macro["monotone_risk_invariant_violations"])
    for class_payload in per_class.values():
        violations.extend(class_payload["monotone_risk_invariant_violations"])
    return violations


def _scaled_thresholds(tau_lo: float, tau_hi: float, width_factor: float) -> tuple[float, float]:
    scaled_lo = (1.0 - width_factor) * tau_lo + width_factor * 0.5
    scaled_hi = (1.0 - width_factor) * tau_hi + width_factor * 0.5
    return float(scaled_lo), float(scaled_hi)


def _scaled_threshold_arrays(
    thresholds: AbstentionThresholds,
    width_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    tau_lo = (1.0 - width_factor) * thresholds.tau_lo + width_factor * 0.5
    tau_hi = (1.0 - width_factor) * thresholds.tau_hi + width_factor * 0.5
    return tau_lo, tau_hi


def _resolve_classes(config: Mapping[str, Any]) -> list[str]:
    configured = (
        config.get("classes")
        or config.get("data", {}).get("labels")
        or config.get("abstention", {}).get("classes")
    )
    if not configured:
        raise ValueError("Abstention thresholds require an explicit ordered class list.")
    return list(configured)


def _resolve_curve_class_names(class_names: list[str] | None, expected_classes: int) -> list[str]:
    if class_names is None:
        return [f"class_{index}" for index in range(expected_classes)]
    if len(class_names) != expected_classes:
        raise ValueError("class_names length must match probability columns.")
    return list(class_names)


def _validate_probs(probs: np.ndarray, expected_classes: int) -> np.ndarray:
    probabilities = np.asarray(probs, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != expected_classes:
        raise ValueError(f"Expected probabilities with shape [N, {expected_classes}].")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must be in [0, 1].")
    return probabilities


def _validate_threshold_array(values: np.ndarray, name: str, expected_classes: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != expected_classes:
        raise ValueError(f"{name} must be a per-class array of length {expected_classes}.")
    if np.any((array <= 0.0) | (array >= 1.0)):
        raise ValueError(f"{name} values must be inside (0, 1).")
    return array
