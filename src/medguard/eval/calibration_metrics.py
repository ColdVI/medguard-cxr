"""Calibration metrics and reliability diagrams for Phase 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PHASE = "2"


def is_available() -> bool:
    """Return whether calibration metrics are implemented."""
    return True


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    binning: str = "equal_width",
) -> float | np.ndarray:
    """Compute ECE for one class or per class for ``[N, C]`` arrays."""
    probs, labels = _validate_prob_label_shapes(probs, labels)
    if probs.ndim == 1:
        return _ece_1d(probs, labels, n_bins=n_bins, binning=binning)
    return np.asarray(
        [
            _ece_1d(probs[:, index], labels[:, index], n_bins=n_bins, binning=binning)
            for index in range(probs.shape[1])
        ],
        dtype=np.float64,
    )


def maximum_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    binning: str = "equal_width",
) -> float | np.ndarray:
    """Compute MCE for one class or per class for ``[N, C]`` arrays."""
    probs, labels = _validate_prob_label_shapes(probs, labels)
    if probs.ndim == 1:
        return _mce_1d(probs, labels, n_bins=n_bins, binning=binning)
    return np.asarray(
        [
            _mce_1d(probs[:, index], labels[:, index], n_bins=n_bins, binning=binning)
            for index in range(probs.shape[1])
        ],
        dtype=np.float64,
    )


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float | np.ndarray:
    """Return mean squared probability error for one or many classes."""
    probs, labels = _validate_prob_label_shapes(probs, labels)
    squared = (probs - labels) ** 2
    if probs.ndim == 1:
        return float(np.mean(squared))
    return np.mean(squared, axis=0)


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(bin_centers, bin_acc, bin_count)`` for one class."""
    probs, labels = _validate_prob_label_shapes(probs, labels)
    if probs.ndim != 1:
        raise ValueError("reliability_curve expects one-dimensional inputs.")
    bins = _bin_indices(probs, n_bins=n_bins, binning="equal_width")
    centers = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
    accuracies = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    for bin_index in range(n_bins):
        mask = bins == bin_index
        counts[bin_index] = float(np.sum(mask))
        if np.any(mask):
            accuracies[bin_index] = float(np.mean(labels[mask]))
    return centers, accuracies, counts


def plot_reliability_diagram(
    probs_pre: np.ndarray,
    probs_post: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    n_bins: int = 15,
    is_synthetic: bool = False,
) -> None:
    """Write a 4x4 reliability diagram PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs_pre, labels = _validate_prob_label_shapes(probs_pre, labels)
    probs_post, labels = _validate_prob_label_shapes(probs_post, labels)
    if probs_pre.ndim != 2:
        raise ValueError("Reliability diagrams expect [N, C] probability arrays.")
    if probs_pre.shape[1] != len(class_names):
        raise ValueError("class_names length must match probability columns.")

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    flat_axes = axes.flatten()

    for class_index, class_name in enumerate(class_names):
        axis = flat_axes[class_index]
        _plot_one_axis(
            axis=axis,
            probs_pre=probs_pre[:, class_index],
            probs_post=probs_post[:, class_index],
            labels=labels[:, class_index],
            title_name=class_name,
            n_bins=n_bins,
            row=class_index // 4,
            col=class_index % 4,
        )

    macro_axis = flat_axes[len(class_names)]
    _plot_one_axis(
        axis=macro_axis,
        probs_pre=probs_pre.reshape(-1),
        probs_post=probs_post.reshape(-1),
        labels=labels.reshape(-1),
        title_name="Macro pooled",
        n_bins=n_bins,
        row=len(class_names) // 4,
        col=len(class_names) % 4,
    )

    legend_axis = flat_axes[-1]
    legend_axis.axis("off")
    handles, labels_for_legend = flat_axes[0].get_legend_handles_labels()
    legend_axis.legend(handles, labels_for_legend, loc="center")

    if is_synthetic:
        fig.suptitle(
            "SYNTHETIC SMOKE — NOT A CLINICAL EVALUATION",
            color="red",
            fontsize=18,
            fontweight="bold",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.97 if is_synthetic else 1))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _plot_one_axis(
    axis: object,
    probs_pre: np.ndarray,
    probs_post: np.ndarray,
    labels: np.ndarray,
    title_name: str,
    n_bins: int,
    row: int,
    col: int,
) -> None:
    centers, acc_pre, count_pre = reliability_curve(probs_pre, labels, n_bins=n_bins)
    _, acc_post, count_post = reliability_curve(probs_post, labels, n_bins=n_bins)
    ece_pre = expected_calibration_error(probs_pre, labels, n_bins=n_bins)
    ece_post = expected_calibration_error(probs_post, labels, n_bins=n_bins)

    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="ideal")
    axis.scatter(
        centers,
        acc_pre,
        s=np.maximum(count_pre, 1.0) * 3.0,
        color="red",
        alpha=0.75,
        label="uncalibrated",
    )
    axis.plot(centers, acc_pre, color="red", alpha=0.75)
    axis.scatter(
        centers,
        acc_post,
        s=np.maximum(count_post, 1.0) * 3.0,
        color="blue",
        alpha=0.75,
        label="calibrated",
    )
    axis.plot(centers, acc_post, color="blue", alpha=0.75)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title(f"{title_name} ECE pre->post: {ece_pre:.3f} -> {ece_post:.3f}")
    if row == 3:
        axis.set_xlabel("Confidence")
    if col == 0:
        axis.set_ylabel("Empirical positive rate")


def _ece_1d(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
    binning: str,
) -> float:
    bins = _bin_indices(probs, n_bins=n_bins, binning=binning)
    total = float(probs.shape[0])
    if total == 0:
        return 0.0
    error = 0.0
    for bin_index in range(n_bins):
        mask = bins == bin_index
        if not np.any(mask):
            continue
        gap = abs(float(np.mean(labels[mask])) - float(np.mean(probs[mask])))
        error += float(np.sum(mask)) / total * gap
    return float(error)


def _mce_1d(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
    binning: str,
) -> float:
    bins = _bin_indices(probs, n_bins=n_bins, binning=binning)
    max_error = 0.0
    for bin_index in range(n_bins):
        mask = bins == bin_index
        if not np.any(mask):
            continue
        gap = abs(float(np.mean(labels[mask])) - float(np.mean(probs[mask])))
        max_error = max(max_error, gap)
    return float(max_error)


def _bin_indices(probs: np.ndarray, n_bins: int, binning: str) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    if binning == "equal_width":
        return np.minimum((probs * n_bins).astype(int), n_bins - 1)
    if binning == "equal_mass":
        order = np.argsort(probs, kind="mergesort")
        bins = np.empty(probs.shape[0], dtype=int)
        if probs.shape[0] == 0:
            return bins
        bins[order] = np.minimum(
            (np.arange(probs.shape[0]) * n_bins // probs.shape[0]).astype(int),
            n_bins - 1,
        )
        return bins
    raise ValueError(f"Unsupported calibration binning: {binning}")


def _validate_prob_label_shapes(
    probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    validated_probs = np.asarray(probs, dtype=np.float64)
    validated_labels = np.asarray(labels, dtype=np.float64)
    if validated_probs.shape != validated_labels.shape:
        raise ValueError(
            f"Probability shape {validated_probs.shape} does not match label shape "
            f"{validated_labels.shape}."
        )
    if validated_probs.ndim not in {1, 2}:
        raise ValueError("Expected one-dimensional or two-dimensional arrays.")
    return np.clip(validated_probs, 0.0, 1.0), validated_labels
