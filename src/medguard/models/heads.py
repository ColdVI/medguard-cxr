"""Heads and missing-label masking for multi-dataset classification."""

from __future__ import annotations

import torch
from torch import nn


class CanonicalOntologyHead(nn.Module):
    """Shared canonical head with optional dataset-specific auxiliary heads."""

    def __init__(
        self,
        in_features: int,
        canonical_classes: int,
        auxiliary_classes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.canonical = nn.Linear(in_features, canonical_classes)
        self.auxiliary = nn.ModuleDict(
            {
                dataset: nn.Linear(in_features, classes)
                for dataset, classes in (auxiliary_classes or {}).items()
            }
        )

    def forward(
        self,
        features: torch.Tensor,
        dataset: str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        canonical = self.canonical(features)
        if dataset is None or dataset not in self.auxiliary:
            return canonical
        return canonical, self.auxiliary[dataset](features)


def masked_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    """Exclude labels unavailable for a source dataset from pooled loss."""

    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = label_mask.to(dtype=losses.dtype)
    denominator = weights.sum()
    if denominator.item() == 0:
        raise ValueError("A pooled-training batch has no supervised labels")
    return (losses * weights).sum() / denominator
