"""Train the Phase 1 NIH ChestX-ray14 multi-label classifier."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from medguard.data.nih import (
    DatasetUnavailableError,
    NIHChestXray14Dataset,
    compute_pos_weight,
    create_dataloader,
    dataloader_kwargs,
    dataset_available,
)
from medguard.models.classifier import build_classifier, build_loss, probabilities_from_logits


class SyntheticCXRDataset(Dataset[dict[str, Any]]):
    """Deterministic synthetic samples for no-data smoke runs."""

    def __init__(
        self,
        samples: int,
        channels: int,
        image_size: int,
        classes: int,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(samples, channels, image_size, image_size, generator=generator)
        label_rows = []
        for index in range(samples):
            label_rows.append([(index + class_index) % 2 for class_index in range(classes)])
        self.labels = torch.tensor(label_rows, dtype=torch.float32)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "patient_id": f"synthetic-{index}",
            "path": f"synthetic://{index}",
        }

    def labels_tensor(self) -> torch.Tensor:
        """Return all labels as ``[N, C]`` tensor."""
        return self.labels


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_nih.yaml")
    return parser.parse_args()


def main() -> None:
    """Train the classifier or run a documented smoke training path."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 2026)))
    device = resolve_device(config)

    no_data_smoke = not dataset_available(config)
    if no_data_smoke:
        runtime_config = smoke_model_config(config)
        train_loader, val_loader, pos_weight = build_smoke_loaders(runtime_config)
        mode = "smoke_no_dataset"
    else:
        runtime_config = config
        warn_if_random_imagenet_init(runtime_config)
        train_loader, val_loader, pos_weight = build_nih_loaders(runtime_config)
        mode = "nih"

    model = build_classifier(runtime_config).to(device)
    loss_fn = build_loss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("optimizer", {}).get("learning_rate", 1e-4)),
        weight_decay=float(config.get("optimizer", {}).get("weight_decay", 1e-4)),
    )

    train_cfg = config.get("training", {})
    epochs = 1 if no_data_smoke else int(train_cfg.get("epochs", 1))
    patience = int(train_cfg.get("early_stopping", {}).get("patience", 5))
    clip_norm = float(train_cfg.get("gradient_clip_max_norm", 1.0))
    use_amp = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = float("-inf")
    best_report: dict[str, Any] = {}
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            clip_norm=clip_norm,
        )
        val_report = evaluate_loader(model, val_loader, loss_fn, device)
        score = val_report["macro_auroc"]
        score_for_selection = float(score) if score is not None else -float(val_report["loss"])

        if score_for_selection > best_score:
            best_score = score_for_selection
            stale_epochs = 0
            best_report = {"epoch": epoch, "train_loss": train_loss, **val_report}
            save_checkpoint(model, runtime_config, pos_weight, best_report)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    write_training_report(mode=mode, device=device, best_report=best_report, pos_weight=pos_weight)
    print(f"Training completed in {mode} mode. Best epoch: {best_report.get('epoch', 1)}")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config."""
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    """Set reproducibility seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    """Resolve configured device."""
    requested = str(config.get("training", {}).get("device", "auto"))
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_nih_loaders(
    config: Mapping[str, Any],
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]], torch.Tensor]:
    """Build real NIH train/val loaders."""
    try:
        train_dataset = NIHChestXray14Dataset.from_config(config, split="train")
        val_dataset = NIHChestXray14Dataset.from_config(config, split="val")
    except DatasetUnavailableError:
        raise

    pos_weight = compute_pos_weight(train_dataset)
    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    val_loader = create_dataloader(val_dataset, config, shuffle=False)
    return train_loader, val_loader, pos_weight


def build_smoke_loaders(
    config: Mapping[str, Any],
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]], torch.Tensor]:
    """Build deterministic no-data smoke loaders."""
    smoke_cfg = config.get("smoke", {})
    model_cfg = config.get("model", {})
    preprocessing = config.get("preprocessing", {})
    samples = int(smoke_cfg.get("samples", 8))
    batch_size = int(smoke_cfg.get("batch_size", 2))
    classes = int(model_cfg.get("num_classes", 14))
    channels = int(preprocessing.get("channels", 3))
    image_size = int(smoke_cfg.get("image_size", 64))
    seed = int(config.get("seed", 2026))

    dataset = SyntheticCXRDataset(samples, channels, image_size, classes, seed)
    labels = dataset.labels_tensor()
    positive = labels.sum(dim=0)
    negative = labels.shape[0] - positive
    pos_weight = torch.where(
        positive > 0,
        negative / positive.clamp_min(1.0),
        torch.ones_like(positive),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs(config),
    )
    return loader, loader, pos_weight.to(dtype=torch.float32)


def smoke_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a smoke-only runtime config that never downloads pretrained weights."""
    runtime_config = deepcopy(dict(config))
    model_cfg = dict(runtime_config.get("model", {}))
    model_cfg["allow_weight_download"] = False
    runtime_config["model"] = model_cfg
    return runtime_config


def warn_if_random_imagenet_init(config: Mapping[str, Any]) -> None:
    """Warn when real-data training would silently fall back to random initialization."""
    model_cfg = config.get("model", {})
    pretrained = model_cfg.get("pretrained", "imagenet")
    allow_weight_download = bool(model_cfg.get("allow_weight_download", False))
    if pretrained == "imagenet" and not allow_weight_download:
        print(
            "WARNING: Real NIH training is configured with pretrained=imagenet but "
            "allow_weight_download=false; DenseNet121 will be randomly initialized."
        )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    clip_norm: float,
) -> float:
    """Train one epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate raw logits with sigmoid probabilities."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        probs = probabilities_from_logits(logits)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_probs).numpy()
    per_class = []
    for class_index in range(y_true.shape[1]):
        if len(np.unique(y_true[:, class_index])) < 2:
            per_class.append(None)
        else:
            per_class.append(float(roc_auc_score(y_true[:, class_index], y_score[:, class_index])))
    valid_scores = [score for score in per_class if score is not None]
    return {
        "loss": total_loss / max(total_samples, 1),
        "per_class_auroc": per_class,
        "macro_auroc": float(np.mean(valid_scores)) if valid_scores else None,
    }


def save_checkpoint(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    pos_weight: torch.Tensor,
    report: Mapping[str, Any],
) -> None:
    """Save the best checkpoint."""
    checkpoint_path = Path(
        config.get("training", {}).get("checkpoint", {}).get("path", "checkpoints/best.pt")
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "pos_weight": pos_weight.cpu(),
            "report": dict(report),
        },
        checkpoint_path,
    )


def write_training_report(
    mode: str,
    device: torch.device,
    best_report: Mapping[str, Any],
    pos_weight: torch.Tensor,
) -> None:
    """Write a compact training report."""
    output = Path("results/baseline_nih_train.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": mode,
        "device": str(device),
        "best": dict(best_report),
        "pos_weight_source": "computed_from_training_set",
        "pos_weight_formula": "negative_count_c / positive_count_c",
        "pos_weight": [float(value) for value in pos_weight.cpu().tolist()],
        "label_quality": "NIH labels are noisy silver-standard NLP-mined labels.",
        "localization": "not_applicable_nih_image_level_labels_only",
    }
    if mode == "smoke_no_dataset":
        report["WARNING_DO_NOT_USE"] = "synthetic_smoke_only_not_a_real_evaluation"
        report["model_quality_evidence"] = False
        report["smoke_note"] = (
            "Synthetic smoke training exercises code paths only; the best-epoch values "
            "are not model-quality metrics."
        )
    else:
        report["model_quality_evidence"] = True
    output.write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
