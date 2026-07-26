"""Evaluate the Phase 1 NIH ChestX-ray14 multi-label classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from medguard.data.nih import (
    DatasetUnavailableError,
    NIHChestXray14Dataset,
    create_dataloader,
    dataset_available,
)
from medguard.models.classifier import build_classifier, probabilities_from_logits


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_nih.yaml")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Split to evaluate. Use val to produce logits for post-hoc calibration fitting.",
    )
    parser.add_argument(
        "--per-sample-output",
        type=Path,
        help=(
            "Write a per-sample CSV of raw logits and labels. Probabilities are a pure "
            "function of the logits, so any calibrator can be fitted offline from this file."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate the configured checkpoint or run a no-data smoke evaluation."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 2026)))
    labels = list(config.get("data", {}).get("labels", []))

    if dataset_available(config):
        report = evaluate_nih(
            config,
            labels,
            split=args.split,
            per_sample_output=args.per_sample_output,
        )
    else:
        report = evaluate_smoke(config, labels)

    output_path = Path(
        config.get("evaluation", {}).get("output_json", "results/baseline_nih_eval.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote evaluation report to {output_path}")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config."""
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    """Set deterministic seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_nih(
    config: Mapping[str, Any],
    labels: list[str],
    split: str = "test",
    per_sample_output: Path | None = None,
) -> dict[str, Any]:
    """Evaluate on the configured NIH split."""
    try:
        dataset = NIHChestXray14Dataset.from_config(config, split=split)
    except DatasetUnavailableError as exc:
        return evaluate_smoke(config, labels, reason=str(exc))

    batch_size = int(
        config.get("evaluation", {}).get(
            "batch_size",
            config.get("training", {}).get("batch_size", 32),
        )
    )
    loader = create_dataloader(
        dataset,
        config,
        shuffle=False,
        batch_size=batch_size,
    )
    device = resolve_device(config)
    model = build_classifier(config).to(device)
    checkpoint_loaded = load_checkpoint_if_present(model, config, device)
    model.eval()

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    image_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())
            image_ids.extend(batch["image_id"])

    logits_tensor = torch.cat(all_logits)
    y_true = torch.cat(all_labels).numpy()
    y_prob = probabilities_from_logits(logits_tensor).numpy()
    report = {
        "mode": "nih",
        "split": split,
        "checkpoint_loaded": checkpoint_loaded,
        "num_samples": int(y_true.shape[0]),
        **classification_report(y_true, y_prob, labels),
    }
    if per_sample_output is not None:
        write_per_sample_logits(
            output_path=per_sample_output,
            image_ids=image_ids,
            logits=logits_tensor.numpy(),
            y_true=y_true,
            labels=labels,
        )
        report["per_sample_logits"] = str(per_sample_output)
    return report


def write_per_sample_logits(
    output_path: Path,
    image_ids: list[str],
    logits: np.ndarray,
    y_true: np.ndarray,
    labels: list[str],
) -> None:
    """Write raw logits and ground-truth labels for every evaluated image.

    Raw logits are stored rather than probabilities because every calibrator in this
    project is a function of the logit alone: sigmoid, temperature scaling, isotonic and
    Platt can all be fitted or applied offline from this file without another forward
    pass. Storing probabilities instead would discard that.
    """

    if logits.shape != y_true.shape:
        raise ValueError("logits and labels must have the same shape.")
    if len(image_ids) != logits.shape[0]:
        raise ValueError("image_ids must have one entry per evaluated image.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["image_id"]
    header += [f"label_{name}" for name in labels]
    header += [f"logit_{name}" for name in labels]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, image_id in enumerate(image_ids):
            row: list[Any] = [image_id]
            row += [int(value) for value in y_true[index]]
            row += [f"{float(value):.6f}" for value in logits[index]]
            writer.writerow(row)
    print(f"Wrote per-sample logits for {len(image_ids)} images to {output_path}")


def evaluate_smoke(
    config: Mapping[str, Any],
    labels: list[str],
    reason: str = "Configured NIH dataset not found.",
) -> dict[str, Any]:
    """Produce a no-data smoke report without synthetic performance metrics."""
    classes = int(config.get("model", {}).get("num_classes", len(labels) or 14))
    samples = max(4, int(config.get("smoke", {}).get("eval_samples", 8)))
    label_names = labels or [f"class_{index}" for index in range(classes)]
    null_metrics = {
        label: {
            "auroc": None,
            "auprc": None,
            "sensitivity_at_90_specificity": None,
        }
        for label in label_names
    }
    return {
        "WARNING_DO_NOT_USE": "synthetic_smoke_only_not_a_real_evaluation",
        "mode": "smoke_no_dataset",
        "reason": reason,
        "num_samples": int(samples),
        "per_class": null_metrics,
        "macro_auroc": None,
        "macro_auroc_valid_class_count": 0,
        "macro_auprc": None,
        "macro_auprc_valid_class_count": 0,
        "macro_sensitivity_at_90_specificity": None,
        "macro_sensitivity_at_90_specificity_valid_class_count": 0,
        "probability_source": "not_computed_smoke_mode",
        "threshold_tuning": "none",
        "label_quality": "NIH labels are noisy silver-standard NLP-mined labels.",
        "localization": "not_evaluated_nih_image_level_labels_only",
    }


def classification_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Compute Phase 1 classification metrics from probabilities."""
    per_class: dict[str, dict[str, float | None]] = {}
    aurocs: list[float] = []
    auprcs: list[float] = []
    sensitivities: list[float] = []
    label_names = labels or [f"class_{index}" for index in range(y_true.shape[1])]

    for index, label in enumerate(label_names):
        target = y_true[:, index]
        score = y_prob[:, index]
        metrics = {
            "auroc": safe_auroc(target, score),
            "auprc": safe_auprc(target, score),
            "sensitivity_at_90_specificity": sensitivity_at_specificity(
                target,
                score,
                specificity=0.90,
            ),
        }
        per_class[label] = metrics
        if metrics["auroc"] is not None:
            aurocs.append(metrics["auroc"])
        if metrics["auprc"] is not None:
            auprcs.append(metrics["auprc"])
        if metrics["sensitivity_at_90_specificity"] is not None:
            sensitivities.append(metrics["sensitivity_at_90_specificity"])

    return {
        "per_class": per_class,
        "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
        "macro_auroc_valid_class_count": len(aurocs),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
        "macro_auprc_valid_class_count": len(auprcs),
        "macro_sensitivity_at_90_specificity": (
            float(np.mean(sensitivities)) if sensitivities else None
        ),
        "macro_sensitivity_at_90_specificity_valid_class_count": len(sensitivities),
        "probability_source": "torch.sigmoid(raw_logits)",
        "threshold_tuning": "none",
        "label_quality": "NIH labels are noisy silver-standard NLP-mined labels.",
        "localization": "not_evaluated_nih_image_level_labels_only",
    }


def safe_auroc(target: np.ndarray, score: np.ndarray) -> float | None:
    """Compute AUROC when both target classes are present."""
    if len(np.unique(target)) < 2:
        return None
    return float(roc_auc_score(target, score))


def safe_auprc(target: np.ndarray, score: np.ndarray) -> float | None:
    """Compute average precision when at least one positive is present."""
    if float(target.sum()) <= 0:
        return None
    return float(average_precision_score(target, score))


def sensitivity_at_specificity(
    target: np.ndarray,
    score: np.ndarray,
    specificity: float,
) -> float | None:
    """Return max sensitivity with specificity at least the requested value."""
    if len(np.unique(target)) < 2:
        return None
    fpr, tpr, _ = roc_curve(target, score)
    valid = tpr[fpr <= (1.0 - specificity)]
    if valid.size == 0:
        return 0.0
    return float(valid.max())


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    """Resolve configured device."""
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


def load_checkpoint_if_present(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
) -> bool:
    """Load configured checkpoint if it exists."""
    checkpoint_path = Path(
        config.get("training", {}).get("checkpoint", {}).get("path", "checkpoints/best.pt")
    )
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return True


if __name__ == "__main__":
    main()
