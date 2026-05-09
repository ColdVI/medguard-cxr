"""Phase 3 tests for localization metric utilities."""

import numpy as np

from medguard.eval.localization_metrics import (
    average_precision_at_iou,
    average_precision_records_at_iou,
    box_iou,
    cam_to_bbox,
    heatmap_border_fraction,
    heatmap_peak_in_border,
    mean_average_precision_at_iou,
    per_image_recall_at_iou,
    pointing_game_hit,
)


def test_box_iou_known_overlap() -> None:
    """IoU matches a hand-computed overlap."""

    iou = box_iou((0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75))
    assert np.isclose(iou, 1.0 / 7.0)


def test_cam_to_bbox_returns_normalized_extent() -> None:
    """Thresholded CAM support becomes a normalized xyxy box."""

    heatmap = np.zeros((10, 20), dtype=np.float32)
    heatmap[2:6, 4:10] = 1.0

    assert cam_to_bbox(heatmap, threshold=0.5) == (0.2, 0.2, 0.5, 0.6)


def test_pointing_game_hit_checks_argmax_inside_box() -> None:
    """The pointing game checks whether the CAM peak is inside the GT box."""

    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[4, 5] = 1.0

    assert pointing_game_hit(heatmap, (0.4, 0.3, 0.7, 0.6)) is True
    assert pointing_game_hit(heatmap, (0.0, 0.0, 0.2, 0.2)) is False


def test_pointing_game_hit_accepts_multiple_gt_boxes() -> None:
    """The pointing game succeeds when the CAM peak hits any GT box."""

    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[4, 5] = 1.0

    boxes = [(0.0, 0.0, 0.2, 0.2), (0.4, 0.3, 0.7, 0.6)]
    assert pointing_game_hit(heatmap, boxes) is True


def test_heatmap_border_artifact_metrics() -> None:
    """Border artifact helpers flag edge-dominant CAMs for audit reports."""

    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[0, 0] = 1.0
    heatmap[5, 5] = 1.0

    assert np.isclose(heatmap_border_fraction(heatmap, border_fraction=0.2), 0.5)
    assert heatmap_peak_in_border(heatmap, border_fraction=0.2) is True


def test_average_precision_at_iou_matches_single_true_positive() -> None:
    """One high-scoring matched prediction gives AP 1.0."""

    ap = average_precision_at_iou(
        pred_boxes=np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9]]),
        pred_scores=np.array([0.9, 0.1]),
        gt_boxes=np.array([[0.0, 0.0, 0.5, 0.5]]),
        iou_threshold=0.5,
    )

    assert ap == 1.0


def test_mean_average_precision_omits_empty_ground_truth_keys() -> None:
    """mAP aggregates only image/class keys with ground-truth boxes."""

    predictions = [
        {"image_id": "a", "class_name": "Nodule/Mass", "bbox": (0.0, 0.0, 0.5, 0.5), "score": 0.9},
        {"image_id": "b", "class_name": "Mass", "bbox": (0.0, 0.0, 0.5, 0.5), "score": 0.5},
    ]
    ground_truths = [
        {"image_id": "a", "class_name": "Nodule/Mass", "bbox": (0.0, 0.0, 0.5, 0.5)}
    ]

    assert mean_average_precision_at_iou(predictions, ground_truths) == 1.0


def test_mean_average_precision_aggregates_per_class_across_images() -> None:
    """mAP uses a class-level PR curve rather than per-image binary AP."""

    predictions = [
        {"image_id": "a", "class_name": "Nodule/Mass", "bbox": (0.0, 0.0, 0.4, 0.4), "score": 0.9},
        {"image_id": "b", "class_name": "Nodule/Mass", "bbox": (0.6, 0.6, 0.9, 0.9), "score": 0.8},
        {"image_id": "b", "class_name": "Nodule/Mass", "bbox": (0.0, 0.0, 0.4, 0.4), "score": 0.1},
    ]
    ground_truths = [
        {"image_id": "a", "class_name": "Nodule/Mass", "bbox": (0.0, 0.0, 0.4, 0.4)},
        {"image_id": "b", "class_name": "Nodule/Mass", "bbox": (0.0, 0.0, 0.4, 0.4)},
    ]

    assert np.isclose(
        average_precision_records_at_iou(predictions, ground_truths, iou_threshold=0.5),
        5.0 / 6.0,
    )
    assert np.isclose(mean_average_precision_at_iou(predictions, ground_truths), 5.0 / 6.0)
    assert np.isclose(per_image_recall_at_iou(predictions, ground_truths), 1.0)
