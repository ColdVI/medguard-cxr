"""Evaluate Phase 3 Grad-CAM localization on a configured real dataset.

The RSNA path is the active real-image localization audit for NIH Pneumonia
mapped to RSNA Lung Opacity. Reports explicitly mark smoke-trained checkpoints
so outputs are not mistaken for model-quality evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from medguard.data.nih import NIH_LABELS
from medguard.data.rsna import RSNAPneumoniaDataset, dataset_available, read_rsna_image
from medguard.eval.localization_metrics import (
    box_iou,
    cam_to_bbox,
    heatmap_border_fraction,
    heatmap_peak_in_border,
    mean_average_precision_at_iou,
    pointing_game_accuracy,
)
from medguard.explain.gradcam import generate_gradcam
from medguard.explain.overlays import save_overlay, save_overlay_grid
from medguard.models.classifier import build_classifier, probabilities_from_logits

REAL_DATA_BANNER = "REAL RSNA INPUT - NOT A CLINICAL EVALUATION"
SMOKE_CHECKPOINT_WARNING = "real_images_with_smoke_checkpoint_not_model_quality_evidence"


def main(argv: list[str] | None = None) -> int:
    """Run the configured grounding evaluation."""

    args = _parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_name = str(config.get("data", {}).get("dataset", ""))
    if dataset_name != "rsna-pneumonia-detection":
        raise RuntimeError("evaluate_grounding.py currently supports configs/grounding_rsna.yaml.")
    if not dataset_available(config):
        report = _blocked_report(
            config=config,
            checkpoint_path=args.checkpoint,
            reason="RSNA data is not available under the configured data.root.",
        )
    elif not args.checkpoint.exists():
        report = _blocked_report(
            config=config,
            checkpoint_path=args.checkpoint,
            reason=f"Checkpoint not found: {args.checkpoint}",
        )
    else:
        checkpoint = _load_checkpoint(args.checkpoint)
        report = evaluate_rsna_grounding(
            config=config,
            checkpoint=checkpoint,
            checkpoint_path=args.checkpoint,
            max_samples=args.max_samples,
            overlay_count=args.overlay_count,
        )

    output_path = Path(config["localization"]["outputs"]["metrics_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote grounding report to {output_path}")
    return 0


def evaluate_rsna_grounding(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    max_samples: int | None,
    overlay_count: int | None,
) -> dict[str, Any]:
    """Evaluate Grad-CAM localization on RSNA images."""

    localization_cfg = config.get("localization", {})
    evaluation_cfg = localization_cfg.get("evaluation", {})
    gradcam_cfg = config.get("gradcam", {})
    overlay_count = (
        overlay_count if overlay_count is not None else int(evaluation_cfg.get("overlay_count", 20))
    )
    dataset = RSNAPneumoniaDataset.from_config(config)
    max_samples = _resolve_max_samples(
        cli_value=max_samples,
        config_value=evaluation_cfg.get("max_samples", 32),
        dataset_size=len(dataset),
    )
    checkpoint_mode = _checkpoint_mode(checkpoint)
    model_config = dict(checkpoint.get("config", {}))
    model_config.setdefault("model", {})
    model_config["model"]["allow_weight_download"] = False
    model = build_classifier(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_index = NIH_LABELS.index("Pneumonia")
    confidence_gate = float(gradcam_cfg.get("confidence_gate", 0.70))
    cam_threshold = float(gradcam_cfg.get("cam_threshold", 0.60))
    iou_threshold = float(localization_cfg.get("iou_threshold", 0.50))
    method = str(gradcam_cfg.get("method", "gradcam"))
    post_cfg = gradcam_cfg.get("postprocessing", {})
    smoothing_sigma = float(post_cfg.get("smoothing_sigma", 0.0))
    border_suppression_fraction = float(post_cfg.get("border_suppression_fraction", 0.0))
    border_artifact_fraction = float(post_cfg.get("border_artifact_fraction", 0.05))

    overlay_dir = Path(localization_cfg["outputs"]["overlay_dir"])
    overlay_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_rsna_overlays(overlay_dir)
    pred_records: list[dict[str, Any]] = []
    gt_records: list[dict[str, Any]] = []
    heatmaps: list[np.ndarray] = []
    heatmap_gt_boxes: list[list[tuple[float, float, float, float]]] = []
    overlay_images = []
    per_sample: list[dict[str, Any]] = []
    skipped_low_confidence = 0
    skipped_no_cam_box = 0
    y_true: list[int] = []
    y_score: list[float] = []
    best_ious: list[float] = []
    positive_cases_without_cam = 0
    positive_cases_with_cam = 0

    samples = [dataset[index] for index in range(min(max_samples, len(dataset)))]
    for sample in samples:
        image_id = str(sample["image_id"])
        gt_boxes = [tuple(float(value) for value in box) for box in sample["boxes"].tolist()]
        for gt_box in gt_boxes:
            gt_records.append({"image_id": image_id, "class_name": "Lung Opacity", "bbox": gt_box})

        image_tensor = sample["image"].unsqueeze(0)
        with torch.no_grad():
            logits = model(image_tensor)
            probs = probabilities_from_logits(logits)
        confidence = float(probs[0, class_index].item())
        target = int(float(sample["label"].item()) > 0.5)
        y_true.append(target)
        y_score.append(confidence)
        heatmap = generate_gradcam(
            model=model,
            image=sample["image"],
            class_index=class_index,
            confidence=confidence,
            abstained=False,
            abstention_threshold=confidence_gate,
            method=method,
            smoothing_sigma=smoothing_sigma,
            border_suppression_fraction=border_suppression_fraction,
        )
        if heatmap is None:
            skipped_low_confidence += 1
            if target == 1:
                positive_cases_without_cam += 1
            per_sample.append(
                {
                    "image_id": image_id,
                    "confidence": confidence,
                    "generated": False,
                    "reason": "below_confidence_gate",
                }
            )
            continue

        pred_box = cam_to_bbox(heatmap, threshold=cam_threshold)
        if pred_box is None:
            skipped_no_cam_box += 1
            if target == 1:
                positive_cases_without_cam += 1
            per_sample.append(
                {
                    "image_id": image_id,
                    "confidence": confidence,
                    "generated": False,
                    "reason": "no_cam_bbox",
                }
            )
            continue

        pred_records.append(
            {
                "image_id": image_id,
                "class_name": "Lung Opacity",
                "bbox": pred_box,
                "score": confidence,
            }
        )
        if gt_boxes:
            positive_cases_with_cam += 1
            heatmaps.append(heatmap)
            heatmap_gt_boxes.append(gt_boxes)
            best_ious.append(max(box_iou(pred_box, gt_box) for gt_box in gt_boxes))
        border_fraction = heatmap_border_fraction(heatmap, border_fraction=border_artifact_fraction)
        peak_in_border = heatmap_peak_in_border(heatmap, border_fraction=border_artifact_fraction)
        per_sample.append(
            {
                "image_id": image_id,
                "confidence": confidence,
                "generated": True,
                "pred_box": pred_box,
                "gt_box_count": len(gt_boxes),
                "heatmap_border_fraction": border_fraction,
                "heatmap_peak_in_border": peak_in_border,
            }
        )

        if len(overlay_images) < overlay_count:
            original = read_rsna_image(sample["path"])
            overlay_path = overlay_dir / f"rsna_{len(overlay_images):02d}_{image_id}.png"
            save_overlay(
                image=original,
                heatmap=heatmap,
                output_path=overlay_path,
                predicted_box=pred_box,
                ground_truth_box=gt_boxes if gt_boxes else None,
                banner_text=REAL_DATA_BANNER,
            )
            overlay_images.append(read_rsna_image(overlay_path).convert("RGB"))

    grid_path = None
    if overlay_images:
        grid_path = overlay_dir / "rsna_grid.png"
        save_overlay_grid(overlay_images[:16], grid_path, columns=4, banner_text=REAL_DATA_BANNER)

    map_at_iou = mean_average_precision_at_iou(
        predictions=pred_records,
        ground_truths=gt_records,
        iou_threshold=iou_threshold,
    )
    pointing = pointing_game_accuracy(heatmaps, heatmap_gt_boxes) if heatmaps else None
    classification = _binary_classification_report(np.asarray(y_true), np.asarray(y_score))
    localization_metrics = {
        "map_at_0_5": map_at_iou,
        "pointing_game_accuracy": pointing,
        "mean_iou": float(np.mean(best_ious)) if best_ious else None,
        "iou_at_0_5_hit_rate": (
            float(np.mean(np.asarray(best_ious) >= iou_threshold)) if best_ious else None
        ),
        "iou_threshold": iou_threshold,
        "pointing_game_n_generated_positive_cams": positive_cases_with_cam,
        "positive_cases_without_cam_due_to_gate_or_empty_cam": positive_cases_without_cam,
        "note": (
            "CAM-derived weak boxes are extracted from thresholded Grad-CAM support; "
            "the classifier does not directly predict bounding boxes. Pointing game "
            "accuracy is computed only for confidence-gated positive RSNA cases where "
            "a CAM-derived box was generated."
        ),
    }

    warning = _warning_for_checkpoint(checkpoint_mode)
    report: dict[str, Any] = {
        "mode": "rsna_real_images",
        "WARNING_DO_NOT_USE": warning,
        "model_quality_evidence": checkpoint_mode not in {"smoke_no_dataset", "unknown"},
        "reason": _reason(checkpoint_mode),
        "dataset": "rsna-pneumonia-detection",
        "config_path": "configs/grounding_rsna.yaml",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_mode": checkpoint_mode,
        "n_dataset_records": len(dataset),
        "n_evaluated": len(samples),
        "n_ground_truth_boxes": len(gt_records),
        "n_gradcam_generated": len(pred_records),
        "n_classification_positive": int(np.sum(y_true)),
        "n_classification_negative": int(len(y_true) - np.sum(y_true)),
        "skipped_low_confidence": skipped_low_confidence,
        "skipped_no_cam_box": skipped_no_cam_box,
        "confidence_gate": confidence_gate,
        "cam_threshold": cam_threshold,
        "classification": classification,
        "localization": localization_metrics,
        "metrics": {
            "pneumonia_auroc": classification["pneumonia_auroc"],
            "pneumonia_auprc": classification["pneumonia_auprc"],
            "sensitivity_at_90_specificity": classification["sensitivity_at_90_specificity"],
            "map_at_0_5": localization_metrics["map_at_0_5"],
            "pointing_game_accuracy": localization_metrics["pointing_game_accuracy"],
            "mean_iou": localization_metrics["mean_iou"],
        },
        "outputs": {
            "overlay_dir": str(overlay_dir),
            "overlay_grid": str(grid_path) if grid_path else None,
        },
        "per_sample": per_sample,
    }
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/grounding_rsna.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/baseline_nih_best.pt"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overlay-count", type=int)
    return parser.parse_args(argv)


def _resolve_max_samples(cli_value: int | None, config_value: Any, dataset_size: int) -> int:
    if cli_value is not None:
        return min(cli_value, dataset_size)
    if config_value in {None, "all", "ALL"}:
        return dataset_size
    return min(int(config_value), dataset_size)


def _clear_previous_rsna_overlays(overlay_dir: Path) -> None:
    for path in overlay_dir.glob("rsna_*.png"):
        path.unlink()
    grid_path = overlay_dir / "rsna_grid.png"
    if grid_path.exists():
        grid_path.unlink()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def _blocked_report(config: dict[str, Any], checkpoint_path: Path, reason: str) -> dict[str, Any]:
    localization_cfg = config.get("localization", {})
    output_cfg = localization_cfg.get("outputs", {})
    return {
        "mode": "rsna_blocked",
        "WARNING_DO_NOT_USE": "rsna_grounding_evaluation_not_run",
        "model_quality_evidence": False,
        "reason": reason,
        "dataset": "rsna-pneumonia-detection",
        "config_path": "configs/grounding_rsna.yaml",
        "checkpoint_path": str(checkpoint_path),
        "classification": _null_classification_report(),
        "localization": _null_localization_report(),
        "metrics": {
            "pneumonia_auroc": None,
            "pneumonia_auprc": None,
            "sensitivity_at_90_specificity": None,
            "map_at_0_5": None,
            "pointing_game_accuracy": None,
            "mean_iou": None,
        },
        "outputs": {
            "overlay_dir": str(output_cfg.get("overlay_dir", "results/overlays/rsna")),
            "overlay_grid": None,
        },
        "per_sample": [],
    }


def _binary_classification_report(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    return {
        "target": "RSNA Lung Opacity Target",
        "nih_class": "Pneumonia",
        "rsna_label": "Lung Opacity",
        "n": int(y_true.shape[0]),
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(y_true.shape[0] - np.sum(y_true)),
        "pneumonia_auroc": _safe_auroc(y_true, y_score),
        "pneumonia_auprc": _safe_auprc(y_true, y_score),
        "sensitivity_at_90_specificity": _sensitivity_at_specificity(
            y_true,
            y_score,
            specificity=0.90,
        ),
        "probability_source": "torch.sigmoid(raw_logits)[NIH_LABELS.index('Pneumonia')]",
        "threshold_tuning": "none",
        "scope": "NIH Pneumonia probability evaluated against RSNA Lung Opacity only.",
    }


def _null_classification_report() -> dict[str, Any]:
    return {
        "target": "RSNA Lung Opacity Target",
        "nih_class": "Pneumonia",
        "rsna_label": "Lung Opacity",
        "n": 0,
        "n_positive": 0,
        "n_negative": 0,
        "pneumonia_auroc": None,
        "pneumonia_auprc": None,
        "sensitivity_at_90_specificity": None,
        "probability_source": "not_computed",
        "threshold_tuning": "none",
        "scope": "NIH Pneumonia probability evaluated against RSNA Lung Opacity only.",
    }


def _null_localization_report() -> dict[str, Any]:
    return {
        "map_at_0_5": None,
        "pointing_game_accuracy": None,
        "mean_iou": None,
        "iou_at_0_5_hit_rate": None,
        "iou_threshold": None,
        "note": "Localization metrics were not computed because evaluation was blocked.",
    }


def _safe_auroc(target: np.ndarray, score: np.ndarray) -> float | None:
    if target.size == 0 or len(np.unique(target)) < 2:
        return None
    return float(roc_auc_score(target, score))


def _safe_auprc(target: np.ndarray, score: np.ndarray) -> float | None:
    if target.size == 0 or float(np.sum(target)) <= 0.0:
        return None
    return float(average_precision_score(target, score))


def _sensitivity_at_specificity(
    target: np.ndarray,
    score: np.ndarray,
    specificity: float,
) -> float | None:
    if target.size == 0 or len(np.unique(target)) < 2:
        return None
    fpr, tpr, _ = roc_curve(target, score)
    valid = tpr[fpr <= (1.0 - specificity)]
    if valid.size == 0:
        return 0.0
    return float(valid.max())


def _reason(checkpoint_mode: str) -> str:
    if checkpoint_mode == "smoke_no_dataset":
        return (
            "RSNA images and boxes are real, but the checkpoint was trained in smoke mode; "
            "metrics and overlays validate plumbing only, not model quality."
        )
    if checkpoint_mode == "unknown":
        return (
            "RSNA images and boxes are real, but checkpoint provenance is unknown; "
            "do not use metrics as model-quality evidence."
        )
    return "RSNA real-image localization evaluation."


def _warning_for_checkpoint(checkpoint_mode: str) -> str | None:
    if checkpoint_mode == "smoke_no_dataset":
        return SMOKE_CHECKPOINT_WARNING
    if checkpoint_mode == "unknown":
        return "real_images_with_unknown_checkpoint_provenance_not_model_quality_evidence"
    return None


def _checkpoint_mode(checkpoint: dict[str, Any]) -> str:
    report = checkpoint.get("report", {})
    if isinstance(report, dict) and report.get("mode"):
        return str(report["mode"])
    train_report_path = Path("results/baseline_nih_train.json")
    if train_report_path.exists():
        try:
            train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
            if train_report.get("mode"):
                return str(train_report["mode"])
        except json.JSONDecodeError:
            return "unknown"
    return "unknown"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
