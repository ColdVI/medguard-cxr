"""Localization metrics for Phase 3 grounded evaluation.

All boxes are normalized ``xyxy`` coordinates in ``[0, 1]``. These helpers do
not tune thresholds or inspect datasets; callers own split discipline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PHASE = "3"


def is_available() -> bool:
    """Return True once Phase 3 localization helpers are implemented."""

    return True


def box_area(box: Sequence[float]) -> float:
    """Return normalized box area for one ``xyxy`` box."""

    x_min, y_min, x_max, y_max = _validate_box(box)
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU for two normalized ``xyxy`` boxes."""

    ax1, ay1, ax2, ay2 = _validate_box(box_a)
    bx1, by1, bx2, by2 = _validate_box(box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area((ax1, ay1, ax2, ay2)) + box_area((bx1, by1, bx2, by2)) - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def cam_to_bbox(
    heatmap: np.ndarray,
    threshold: float = 0.6,
    min_area_pixels: int = 1,
) -> tuple[float, float, float, float] | None:
    """Convert a CAM heatmap into a normalized ``xyxy`` bounding box.

    Pixels greater than or equal to ``threshold * max(heatmap)`` are retained.
    The return value is ``None`` when no connected support exceeds the threshold.
    """

    heatmap_2d = _validate_heatmap(heatmap)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    if min_area_pixels < 1:
        raise ValueError("min_area_pixels must be >= 1.")

    max_value = float(np.nanmax(heatmap_2d))
    if not np.isfinite(max_value) or max_value <= 0.0:
        return None

    mask = heatmap_2d >= (threshold * max_value)
    ys, xs = np.where(mask)
    if xs.size < min_area_pixels:
        return None

    height, width = heatmap_2d.shape
    x_min = float(xs.min() / width)
    y_min = float(ys.min() / height)
    x_max = float((xs.max() + 1) / width)
    y_max = float((ys.max() + 1) / height)
    return _clip_box((x_min, y_min, x_max, y_max))


def pointing_game_hit(
    heatmap: np.ndarray,
    ground_truth_box: Sequence[float] | Sequence[Sequence[float]],
) -> bool:
    """Return whether the heatmap argmax falls inside any ground-truth box."""

    heatmap_2d = _validate_heatmap(heatmap)
    height, width = heatmap_2d.shape
    flat_index = int(np.nanargmax(heatmap_2d))
    y, x = np.unravel_index(flat_index, heatmap_2d.shape)
    x_norm = (x + 0.5) / width
    y_norm = (y + 0.5) / height
    return any(_point_inside_box(x_norm, y_norm, box) for box in _as_gt_box_list(ground_truth_box))


def average_precision_at_iou(
    pred_boxes: np.ndarray | Sequence[Sequence[float]],
    pred_scores: np.ndarray | Sequence[float],
    gt_boxes: np.ndarray | Sequence[Sequence[float]],
    iou_threshold: float = 0.5,
) -> float | None:
    """Compute AP for one image/class at a fixed IoU threshold.

    Returns ``None`` when there are no ground-truth boxes; callers should omit
    those cases from macro aggregation.
    """

    predictions = _as_box_array(pred_boxes)
    scores = np.asarray(pred_scores, dtype=np.float64)
    ground_truths = _as_box_array(gt_boxes)

    if ground_truths.shape[0] == 0:
        return None
    if predictions.shape[0] == 0:
        return 0.0
    if scores.shape != (predictions.shape[0],):
        raise ValueError("pred_scores must have one score per predicted box.")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1].")

    order = np.argsort(-scores)
    matched_gt: set[int] = set()
    true_positive = np.zeros(order.shape[0], dtype=np.float64)
    false_positive = np.zeros(order.shape[0], dtype=np.float64)

    for rank, pred_index in enumerate(order):
        ious = np.array([box_iou(predictions[pred_index], gt) for gt in ground_truths])
        best_gt = int(np.argmax(ious)) if ious.size else -1
        best_iou = float(ious[best_gt]) if best_gt >= 0 else 0.0
        if best_iou >= iou_threshold and best_gt not in matched_gt:
            true_positive[rank] = 1.0
            matched_gt.add(best_gt)
        else:
            false_positive[rank] = 1.0

    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / max(ground_truths.shape[0], 1)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    return float(_precision_recall_ap(precision, recall))


def mean_average_precision_at_iou(
    predictions: Sequence[Mapping[str, Any]],
    ground_truths: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
) -> float | None:
    """Compute canonical macro mAP@IoU over classes.

    Each prediction record must include ``image_id``, ``class_name``, ``bbox``,
    and ``score``. Each ground-truth record must include ``image_id``,
    ``class_name``, and ``bbox``.
    """

    class_names = sorted({str(item["class_name"]) for item in ground_truths})
    ap_values: list[float] = []
    for class_name in class_names:
        ap = average_precision_records_at_iou(
            [item for item in predictions if str(item["class_name"]) == class_name],
            [item for item in ground_truths if str(item["class_name"]) == class_name],
            iou_threshold=iou_threshold,
        )
        if ap is not None:
            ap_values.append(ap)
    if not ap_values:
        return None
    return float(np.mean(ap_values))


def average_precision_records_at_iou(
    predictions: Sequence[Mapping[str, Any]],
    ground_truths: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
) -> float | None:
    """Compute AP for one class by pooling predictions over all images."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1].")

    gt_by_image: dict[str, list[Sequence[float]]] = {}
    for item in ground_truths:
        gt_by_image.setdefault(str(item["image_id"]), []).append(item["bbox"])
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if total_gt == 0:
        return None

    sorted_predictions = sorted(
        predictions,
        key=lambda item: float(item.get("score", 1.0)),
        reverse=True,
    )
    if not sorted_predictions:
        return 0.0

    matched: dict[str, set[int]] = {image_id: set() for image_id in gt_by_image}
    true_positive = np.zeros(len(sorted_predictions), dtype=np.float64)
    false_positive = np.zeros(len(sorted_predictions), dtype=np.float64)

    for rank, prediction in enumerate(sorted_predictions):
        image_id = str(prediction["image_id"])
        candidate_gt = gt_by_image.get(image_id, [])
        if not candidate_gt:
            false_positive[rank] = 1.0
            continue
        ious = np.array([box_iou(prediction["bbox"], gt_box) for gt_box in candidate_gt])
        best_gt = int(np.argmax(ious))
        if ious[best_gt] >= iou_threshold and best_gt not in matched[image_id]:
            true_positive[rank] = 1.0
            matched[image_id].add(best_gt)
        else:
            false_positive[rank] = 1.0

    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / max(total_gt, 1)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    return float(_precision_recall_ap(precision, recall))


def per_image_recall_at_iou(
    predictions: Sequence[Mapping[str, Any]],
    ground_truths: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
) -> float | None:
    """Diagnostic mean recall over per-image, per-class localization pairs."""

    keys = sorted(
        {
            (str(item["image_id"]), str(item["class_name"]))
            for item in list(predictions) + list(ground_truths)
        }
    )
    recalls: list[float] = []
    for image_id, class_name in keys:
        pred_boxes = [
            item["bbox"]
            for item in predictions
            if str(item["image_id"]) == image_id and str(item["class_name"]) == class_name
        ]
        gt_boxes = [
            item["bbox"]
            for item in ground_truths
            if str(item["image_id"]) == image_id and str(item["class_name"]) == class_name
        ]
        if not gt_boxes:
            continue
        matched_gt: set[int] = set()
        for pred_box in pred_boxes:
            ious = np.array([box_iou(pred_box, gt_box) for gt_box in gt_boxes])
            if ious.size == 0:
                continue
            best_gt = int(np.argmax(ious))
            if ious[best_gt] >= iou_threshold:
                matched_gt.add(best_gt)
        recalls.append(len(matched_gt) / len(gt_boxes))
    if not recalls:
        return None
    return float(np.mean(recalls))


def pointing_game_accuracy(
    heatmaps: Sequence[np.ndarray],
    gt_boxes: Sequence[Sequence[float] | Sequence[Sequence[float]]],
) -> float | None:
    """Compute pointing-game accuracy over paired CAM heatmaps and boxes."""

    if len(heatmaps) != len(gt_boxes):
        raise ValueError("heatmaps and gt_boxes must have the same length.")
    if not heatmaps:
        return None
    hits = [
        pointing_game_hit(heatmap, box)
        for heatmap, box in zip(heatmaps, gt_boxes, strict=True)
    ]
    return float(np.mean(hits))


def iou_summary(
    pred_boxes: Sequence[Sequence[float] | None],
    gt_boxes: Sequence[Sequence[float]],
) -> dict[str, float | None]:
    """Summarize paired predicted/ground-truth IoU values."""

    if len(pred_boxes) != len(gt_boxes):
        raise ValueError("pred_boxes and gt_boxes must have the same length.")
    values = [
        box_iou(pred, gt)
        for pred, gt in zip(pred_boxes, gt_boxes, strict=True)
        if pred is not None
    ]
    if not values:
        return {"mean_iou": None, "median_iou": None, "n": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_iou": float(np.mean(arr)),
        "median_iou": float(np.median(arr)),
        "n": float(arr.size),
    }


def _precision_recall_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    padded_precision = np.concatenate(([0.0], precision, [0.0]))
    padded_recall = np.concatenate(([0.0], recall, [1.0]))
    for index in range(padded_precision.size - 2, -1, -1):
        padded_precision[index] = max(padded_precision[index], padded_precision[index + 1])
    changed = np.where(padded_recall[1:] != padded_recall[:-1])[0]
    ap_terms = (padded_recall[changed + 1] - padded_recall[changed]) * padded_precision[changed + 1]
    return float(np.sum(ap_terms))


def _as_box_array(boxes: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4].")
    for box in arr:
        _validate_box(box)
    return arr


def _as_gt_box_list(
    boxes: Sequence[float] | Sequence[Sequence[float]],
) -> list[tuple[float, float, float, float]]:
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.shape == (4,):
        return [_validate_box(arr)]
    if arr.ndim == 2 and arr.shape[1] == 4:
        return [_validate_box(box) for box in arr]
    raise ValueError("ground_truth_box must be one box or an array of boxes with shape [N, 4].")


def _point_inside_box(x: float, y: float, box: Sequence[float]) -> bool:
    x_min, y_min, x_max, y_max = _validate_box(box)
    return bool(x_min <= x <= x_max and y_min <= y <= y_max)


def _validate_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(box, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError("box must contain exactly four xyxy values.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("box coordinates must be finite.")
    x_min, y_min, x_max, y_max = (_clip_float(value) for value in arr)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("box must satisfy x_max > x_min and y_max > y_min.")
    return x_min, y_min, x_max, y_max


def _clip_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = (_clip_float(value) for value in box)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("CAM-derived box collapsed after clipping.")
    return x_min, y_min, x_max, y_max


def _clip_float(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _validate_heatmap(heatmap: np.ndarray) -> np.ndarray:
    arr = np.asarray(heatmap, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("heatmap must be a 2D array.")
    if arr.size == 0:
        raise ValueError("heatmap must not be empty.")
    if not np.isfinite(arr).any():
        raise ValueError("heatmap must contain at least one finite value.")
    return arr
