"""Tests for the per-sample logit dump written by the evaluation script."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.evaluate import write_per_sample_logits

from medguard.models.classifier import probabilities_from_logits

LABELS = ["Atelectasis", "Cardiomegaly", "Effusion"]


def read_dump(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read the dump back into image ids, labels and logits."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    image_ids = [row["image_id"] for row in rows]
    labels = np.array([[float(row[f"label_{name}"]) for name in LABELS] for row in rows])
    logits = np.array([[float(row[f"logit_{name}"]) for name in LABELS] for row in rows])
    return image_ids, labels, logits


def test_dump_is_sufficient_to_reconstruct_probabilities(tmp_path: Path) -> None:
    """Sigmoid of the stored logits reproduces the evaluated probabilities.

    This is the property that makes the artifact useful: any post-hoc calibrator can be
    fitted from it offline, with no second forward pass through the network.
    """

    rng = np.random.default_rng(2026)
    logits = rng.normal(0.0, 3.0, size=(64, len(LABELS)))
    y_true = (rng.random((64, len(LABELS))) < 0.2).astype(float)
    image_ids = [f"img_{index:04d}.png" for index in range(64)]
    output = tmp_path / "per_sample.csv"

    write_per_sample_logits(
        output_path=output,
        image_ids=image_ids,
        logits=logits,
        y_true=y_true,
        labels=LABELS,
    )
    read_ids, read_labels, read_logits = read_dump(output)

    expected = probabilities_from_logits(torch.from_numpy(logits)).numpy()
    recovered = probabilities_from_logits(torch.from_numpy(read_logits)).numpy()

    assert read_ids == image_ids
    assert np.array_equal(read_labels, y_true)
    assert np.allclose(recovered, expected, atol=1e-6)


def test_dump_preserves_row_order_and_prevalence(tmp_path: Path) -> None:
    """Row order matches evaluation order and per-class positive counts survive."""

    logits = np.array([[1.0, -2.0, 0.5], [-1.0, 3.0, 0.0], [0.25, 0.25, -4.0]])
    y_true = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    image_ids = ["b.png", "a.png", "c.png"]
    output = tmp_path / "per_sample.csv"

    write_per_sample_logits(
        output_path=output,
        image_ids=image_ids,
        logits=logits,
        y_true=y_true,
        labels=LABELS,
    )
    read_ids, read_labels, _ = read_dump(output)

    assert read_ids == image_ids
    assert read_labels.sum(axis=0).tolist() == y_true.sum(axis=0).tolist()


def test_dump_rejects_mismatched_inputs(tmp_path: Path) -> None:
    """Shape mismatches fail loudly instead of writing a misaligned artifact."""

    logits = np.zeros((4, len(LABELS)))
    y_true = np.zeros((4, len(LABELS)))

    with pytest.raises(ValueError, match="one entry per evaluated image"):
        write_per_sample_logits(
            output_path=tmp_path / "a.csv",
            image_ids=["only_one.png"],
            logits=logits,
            y_true=y_true,
            labels=LABELS,
        )

    with pytest.raises(ValueError, match="same shape"):
        write_per_sample_logits(
            output_path=tmp_path / "b.csv",
            image_ids=[f"{index}.png" for index in range(4)],
            logits=logits,
            y_true=np.zeros((4, len(LABELS) + 1)),
            labels=LABELS,
        )
