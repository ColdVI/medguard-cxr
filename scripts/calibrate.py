"""Fit Phase 2 calibrators and evaluate calibrated selective prediction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from medguard.data.nih import NIHChestXray14Dataset, create_dataloader, dataset_available
from medguard.eval.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    plot_reliability_diagram,
)
from medguard.models.calibration import (
    Calibrator,
    build_calibrator,
    hash_image_ids,
    save_calibrator,
)
from medguard.models.classifier import build_classifier, probabilities_from_logits
from medguard.safety.abstention import (
    load_thresholds_from_config,
    policy_selective_risk_curve,
    selective_risk_curve,
)


@dataclass
class SplitPredictions:
    """Cached logits, labels, and image IDs for one split."""

    logits: np.ndarray
    labels: np.ndarray
    image_ids: list[str]


def parse_args() -> argparse.Namespace:
    """Parse calibration CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/calibration.yaml")
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--method",
        choices=["temperature", "isotonic", "platt", "all"],
        default=None,
    )
    parser.add_argument("--force-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run calibration from CLI arguments."""
    run_calibration(parse_args())


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full Phase 2 calibration pipeline and return the report."""
    config_path = Path(args.config)
    calibration_config = load_yaml(config_path)
    _assert_fit_split_is_val(calibration_config)

    baseline_path = Path(args.baseline_config or calibration_config["baseline_config"])
    baseline_config = load_yaml(baseline_path)
    classes = list(baseline_config.get("data", {}).get("labels", []))
    if not classes:
        raise RuntimeError("Baseline config must provide ordered NIH class labels.")

    method = args.method or calibration_config.get("calibration", {}).get("method", "temperature")
    methods = _methods_to_run(method)
    checkpoint_path = Path(
        args.checkpoint
        or baseline_config.get("training", {}).get("checkpoint", {}).get(
            "path",
            "checkpoints/baseline_nih_best.pt",
        )
    )
    mode, reason = resolve_mode(args.force_smoke, baseline_config, checkpoint_path)

    if mode == "nih":
        val_predictions, test_predictions = collect_nih_predictions(
            baseline_config=baseline_config,
            checkpoint_path=checkpoint_path,
        )
    else:
        val_predictions, test_predictions = synthetic_predictions(calibration_config, classes)

    n_bins = int(calibration_config.get("calibration", {}).get("n_bins", 15))
    binning = str(calibration_config.get("calibration", {}).get("binning", "equal_width"))
    val_hash = hash_image_ids(val_predictions.image_ids)
    test_hash = hash_image_ids(test_predictions.image_ids)

    probs_pre_val = _sigmoid_np(val_predictions.logits)
    probs_pre_test = _sigmoid_np(test_predictions.logits)
    validation_uncalibrated = metric_block(
        probs=probs_pre_val,
        labels=val_predictions.labels,
        classes=classes,
        n_bins=n_bins,
        binning=binning,
    )
    uncalibrated = metric_block(
        probs=probs_pre_test,
        labels=test_predictions.labels,
        classes=classes,
        n_bins=n_bins,
        binning=binning,
    )

    calibrated: dict[str, Any] = {}
    validation_calibrated: dict[str, Any] = {}
    improvement: dict[str, Any] = {}
    calibrators: dict[str, Calibrator] = {}
    for method_name in methods:
        calibrator = build_calibrator(method_name, classes)
        calibrator.fit(
            val_predictions.logits,
            val_predictions.labels,
            image_ids=val_predictions.image_ids,
        )
        calibrators[method_name] = calibrator
        validation_calibrated[method_name] = {
            **metric_block(
                probs=calibrator.transform(val_predictions.logits),
                labels=val_predictions.labels,
                classes=classes,
                n_bins=n_bins,
                binning=binning,
            ),
            **_calibrator_report_fields(calibrator),
        }
        probs_post_test = calibrator.transform(test_predictions.logits)
        calibrated_block = metric_block(
            probs=probs_post_test,
            labels=test_predictions.labels,
            classes=classes,
            n_bins=n_bins,
            binning=binning,
        )
        calibrated[method_name] = {
            **calibrated_block,
            **_calibrator_report_fields(calibrator),
        }
        improvement[method_name] = improvement_block(uncalibrated, calibrated_block, classes)

    temperature = calibrators["temperature"]
    temperature_probs_test = temperature.transform(test_predictions.logits)
    thresholds = load_thresholds_from_config({**calibration_config, "classes": classes})
    selective_risk = selective_risk_curve(
        temperature_probs_test,
        test_predictions.labels,
        n_points=int(calibration_config.get("abstention", {}).get("selective_risk_n_points", 21)),
        class_names=classes,
    )
    policy_selective_risk = policy_selective_risk_curve(
        temperature_probs_test,
        test_predictions.labels,
        thresholds=thresholds,
        n_points=int(calibration_config.get("abstention", {}).get("selective_risk_n_points", 21)),
    )
    invariant_passed, invariant_violations = _risk_invariant_summary(
        {
            "selective_risk": selective_risk,
            "policy_selective_risk": policy_selective_risk,
        }
    )

    output_cfg = calibration_config.get("calibration", {}).get("output", {})
    save_calibrator(temperature, output_cfg.get("pickle", "calibrators/nih_temp_scaling.pkl"))
    for method_name, calibrator in calibrators.items():
        if method_name == "temperature":
            continue
        save_calibrator(calibrator, _secondary_calibrator_path(method_name))

    diagram_path = Path(output_cfg.get("diagram", "results/reliability_diagram.png"))
    plot_reliability_diagram(
        probs_pre=probs_pre_test,
        probs_post=temperature_probs_test,
        labels=test_predictions.labels,
        class_names=classes,
        save_path=diagram_path,
        n_bins=n_bins,
        is_synthetic=mode == "smoke_no_dataset",
    )

    report: dict[str, Any] = {
        "mode": mode,
        "reason": reason,
        "config_paths": {
            "calibration": str(config_path),
            "baseline": str(baseline_path),
            "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else None,
        },
        "split_metadata": {
            "val_n": int(val_predictions.labels.shape[0]),
            "test_n": int(test_predictions.labels.shape[0]),
            "val_id_hash": val_hash,
            "test_id_hash": test_hash,
            "fit_split_hash": temperature.fit_split_hash,
            "no_test_leakage_assertion": (
                "val_id_hash != test_id_hash AND fit_split_hash == val_id_hash"
            ),
        },
        "uncalibrated": uncalibrated,
        "validation_uncalibrated": validation_uncalibrated,
        "calibrated": calibrated,
        "validation_calibrated": validation_calibrated,
        "improvement": improvement,
        "abstention": {
            "thresholds": _thresholds_to_report(calibration_config),
            "selective_risk": selective_risk,
            "policy_selective_risk": policy_selective_risk,
            "monotone_risk_invariant_passed": invariant_passed,
            "monotone_risk_invariant_violations": invariant_violations,
        },
    }
    if mode == "smoke_no_dataset":
        report = {
            "WARNING_DO_NOT_USE": "synthetic_smoke_only_not_a_real_evaluation",
            "smoke_calibration_note": (
                "Synthetic smoke data exercises calibration and abstention mechanics only; "
                "macro deltas are not meaningful model-quality evidence."
            ),
            **report,
        }

    sanitized = sanitize_for_json(report)
    report_path = Path(output_cfg.get("report", "results/calibration_report.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(sanitized, indent=2, allow_nan=False))
    print(f"Wrote calibration report to {report_path}")
    print(f"Wrote reliability diagram to {diagram_path}")
    return sanitized


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file."""
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def resolve_mode(
    force_smoke: bool,
    baseline_config: Mapping[str, Any],
    checkpoint_path: Path,
) -> tuple[str, str]:
    """Resolve operational mode and human-readable reason."""
    if force_smoke:
        return "smoke_no_dataset", "--force-smoke was set."
    data_ready = dataset_available(baseline_config)
    checkpoint_ready = checkpoint_path.exists()
    if data_ready and checkpoint_ready:
        return "nih", "NIH dataset and checkpoint are available."
    if data_ready and not checkpoint_ready:
        raise FileNotFoundError(
            f"NIH dataset is available but checkpoint is missing: {checkpoint_path}. "
            "Run Phase 1 training before real calibration or use --force-smoke."
        )
    return "smoke_no_dataset", "Configured NIH dataset not found."


@torch.no_grad()
def collect_nih_predictions(
    baseline_config: Mapping[str, Any],
    checkpoint_path: Path,
) -> tuple[SplitPredictions, SplitPredictions]:
    """Collect val/test logits from a saved Phase 1 checkpoint."""
    runtime_config = dict(baseline_config)
    runtime_model_cfg = dict(runtime_config.get("model", {}))
    # The checkpoint provides trained weights; keep rebuilds network-independent.
    runtime_model_cfg["allow_weight_download"] = False
    runtime_config["model"] = runtime_model_cfg
    device = _resolve_device(runtime_config)
    model = build_classifier(runtime_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_dataset = NIHChestXray14Dataset.from_config(baseline_config, split="val")
    test_dataset = NIHChestXray14Dataset.from_config(baseline_config, split="test")
    val_loader = create_dataloader(val_dataset, baseline_config, shuffle=False)
    test_loader = create_dataloader(test_dataset, baseline_config, shuffle=False)
    return (
        _collect_loader_predictions(model, val_loader, device),
        _collect_loader_predictions(model, test_loader, device),
    )


@torch.no_grad()
def _collect_loader_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader[dict[str, Any]],
    device: torch.device,
) -> SplitPredictions:
    logits_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    image_ids: list[str] = []
    for batch in loader:
        logits = model(batch["image"].to(device))
        logits_rows.append(logits.cpu())
        label_rows.append(batch["label"].cpu())
        image_ids.extend(_batch_image_ids(batch))
    return SplitPredictions(
        logits=torch.cat(logits_rows).numpy(),
        labels=torch.cat(label_rows).numpy(),
        image_ids=image_ids,
    )


def synthetic_predictions(
    config: Mapping[str, Any],
    classes: list[str],
) -> tuple[SplitPredictions, SplitPredictions]:
    """Generate deterministic, non-perfect synthetic logits for smoke mode."""
    seed = int(config.get("seed", 2026))
    rng = np.random.default_rng(seed)
    smoke_cfg = config.get("smoke", {})
    val_n = int(smoke_cfg.get("val_samples", 768))
    test_n = int(smoke_cfg.get("test_samples", 768))
    alpha = float(smoke_cfg.get("alpha", 0.5))
    prevalence = _smoke_prevalence(config, classes)

    def one_split(name: str, samples: int) -> SplitPredictions:
        labels = rng.binomial(1, prevalence.reshape(1, -1), size=(samples, len(classes)))
        noise = rng.normal(0.0, 1.0, size=(samples, len(classes)))
        logits = alpha * (2.0 * labels - 1.0) + noise
        return SplitPredictions(
            logits=logits.astype(np.float64),
            labels=labels.astype(np.float64),
            image_ids=[f"synthetic-{name}-{index:05d}" for index in range(samples)],
        )

    return one_split("val", val_n), one_split("test", test_n)


def metric_block(
    probs: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    n_bins: int,
    binning: str,
) -> dict[str, Any]:
    """Build per-class and macro ECE/MCE/Brier metrics."""
    ece = np.asarray(expected_calibration_error(probs, labels, n_bins=n_bins, binning=binning))
    mce = np.asarray(maximum_calibration_error(probs, labels, n_bins=n_bins, binning=binning))
    brier = np.asarray(brier_score(probs, labels))
    per_class: dict[str, dict[str, float | None]] = {}

    for class_index, class_name in enumerate(classes):
        target = labels[:, class_index]
        if np.unique(target).size < 2:
            per_class[class_name] = {"ece": None, "mce": None, "brier": None}
            continue
        per_class[class_name] = {
            "ece": float(ece[class_index]),
            "mce": float(mce[class_index]),
            "brier": float(brier[class_index]),
        }

    return {
        "per_class": per_class,
        "macro_ece": _macro_metric(per_class, "ece"),
        "macro_mce": _macro_metric(per_class, "mce"),
        "macro_brier": _macro_metric(per_class, "brier"),
    }


def improvement_block(
    uncalibrated: Mapping[str, Any],
    calibrated: Mapping[str, Any],
    classes: list[str],
) -> dict[str, Any]:
    """Compare calibrated metrics against uncalibrated metrics."""
    improved_names: list[str] = []
    regressed_names: list[str] = []
    for class_name in classes:
        pre = uncalibrated["per_class"][class_name]["ece"]
        post = calibrated["per_class"][class_name]["ece"]
        if pre is None or post is None:
            continue
        if post < pre:
            improved_names.append(class_name)
        elif post > pre:
            regressed_names.append(class_name)
    return {
        "macro_ece_delta": _metric_delta(uncalibrated, calibrated, "macro_ece"),
        "macro_mce_delta": _metric_delta(uncalibrated, calibrated, "macro_mce"),
        "macro_brier_delta": _metric_delta(uncalibrated, calibrated, "macro_brier"),
        "improved_classes": len(improved_names),
        "regressed_classes": len(regressed_names),
        "improved_class_names": improved_names,
        "regressed_class_names": regressed_names,
    }


def _metric_delta(
    uncalibrated: Mapping[str, Any],
    calibrated: Mapping[str, Any],
    metric_name: str,
) -> float | None:
    pre = uncalibrated.get(metric_name)
    post = calibrated.get(metric_name)
    return None if pre is None or post is None else float(pre - post)


def sanitize_for_json(value: Any) -> Any:
    """Replace NaN/Inf values with ``None`` and convert NumPy scalars."""
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _assert_fit_split_is_val(config: Mapping[str, Any]) -> None:
    fit_split = config.get("calibration", {}).get("fit_split", "val")
    if fit_split != "val":
        raise RuntimeError(
            "Calibration must fit on val only — Section 4 critical implementation note."
        )


def _methods_to_run(method: str) -> list[str]:
    if method == "all":
        return ["temperature", "isotonic", "platt"]
    if method == "temperature":
        return ["temperature"]
    return ["temperature", method]


def _calibrator_report_fields(calibrator: Calibrator) -> dict[str, Any]:
    payload = calibrator.to_dict()
    method = payload["method"]
    if method == "temperature":
        return {
            "temperatures": payload["temperatures"],
            "degenerate_classes": payload["degenerate_classes"],
            "fit_split_hash": payload["fit_split_hash"],
        }
    if method == "isotonic":
        return {
            "isotonic_n_thresholds": payload["isotonic_n_thresholds"],
            "degenerate_classes": payload["degenerate_classes"],
            "fit_split_hash": payload["fit_split_hash"],
        }
    return {
        "A": payload["A"],
        "B": payload["B"],
        "degenerate_classes": payload["degenerate_classes"],
        "fit_split_hash": payload["fit_split_hash"],
    }


def _secondary_calibrator_path(method: str) -> Path:
    return Path("calibrators") / f"nih_{method}.pkl"


def _thresholds_to_report(config: Mapping[str, Any]) -> dict[str, Any]:
    abstention_cfg = config.get("abstention", {})
    return {
        "default": abstention_cfg.get("default", {}),
        "rare": abstention_cfg.get("rare_classes", {}),
    }


def _risk_invariant_summary(curves: Mapping[str, Mapping[str, Any]]) -> tuple[bool, list[dict]]:
    violations = []
    for curve_name, curve in curves.items():
        for violation in _risk_invariant_violations(curve):
            violations.append({"curve": curve_name, **violation})
    return not violations, violations


def _risk_invariant_violations(curve: Mapping[str, Any]) -> list[dict]:
    violations = list(curve["macro"].get("monotone_risk_invariant_violations", []))
    for class_payload in curve["per_class"].values():
        violations.extend(class_payload.get("monotone_risk_invariant_violations", []))
    return violations


def _macro_metric(per_class: Mapping[str, Mapping[str, float | None]], name: str) -> float | None:
    values = [metrics[name] for metrics in per_class.values() if metrics[name] is not None]
    return None if not values else float(np.mean(values))


def _batch_image_ids(batch: Mapping[str, Any]) -> list[str]:
    paths = batch.get("path")
    if paths is None:
        return [f"unknown-{index}" for index in range(len(batch["label"]))]
    return [Path(str(path)).name for path in paths]


def _smoke_prevalence(config: Mapping[str, Any], classes: list[str]) -> np.ndarray:
    prevalence_cfg = config.get("smoke", {}).get("prevalence", {})
    default = float(prevalence_cfg.get("default", 0.05))
    common_value = float(prevalence_cfg.get("common_value", 0.20))
    common = set(prevalence_cfg.get("common", []))
    prevalence = np.full(len(classes), default, dtype=np.float64)
    for index, class_name in enumerate(classes):
        if class_name in common:
            prevalence[index] = common_value
    return prevalence


def _resolve_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(
        config.get("evaluation", {}).get(
            "device",
            config.get("training", {}).get("device", "auto"),
        )
    )
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return probabilities_from_logits(torch.as_tensor(logits, dtype=torch.float32)).numpy()


if __name__ == "__main__":
    main()
