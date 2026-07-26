# Grad-CAM Visual Review — RSNA Lung Opacity

Qualitative review of the Grad-CAM overlays produced by the cross-dataset grounding run.
It records what the heatmaps look like and why the quantitative localization metrics come
out the way they do. It is an engineering review of saliency behaviour, not a clinical
assessment.

## Provenance

| Field | Value |
|---|---|
| Artifact | `results/grounding_rsna_eval.json` |
| Overlays | `results/overlays/rsna/` (20 PNGs + `rsna_grid.png`) |
| Checkpoint | `checkpoints/baseline_nih_best.pt` (`checkpoint_mode: nih`) |
| Config | `configs/grounding_rsna.yaml` |
| Provenance flags | `WARNING_DO_NOT_USE: null`, `model_quality_evidence: true` |

The overlays come from the trained NIH checkpoint, not from a smoke checkpoint. Label
scope is narrow: the NIH `Pneumonia` probability evaluated against the RSNA
`Lung Opacity` boxes, and nothing else.

## Quantitative anchors

**Updated after the threshold sweep below.** These anchors now reflect the
val-selected `cam_threshold = 0.80`, read once on the RSNA **test** split (the
`split.eval: test` confirmatory run called for in "Open items"). The earlier anchors —
`cam_threshold = 0.60` on the **val** split — were AUROC 0.8077, AUPRC 0.5595,
pointing game 0.5138, mean IoU 0.2551, mAP@0.5 0.00042; they remain the row of the
sweep table below at threshold 0.60 and are not repeated here as the current artifact.

Classification on the RSNA test subset (n = 1024, 237 positive):
AUROC 0.8277, AUPRC 0.5655, sensitivity at 90% specificity 0.4937.

Localization:

| Metric | Value |
|---|---|
| Pointing game | 0.4286 (n = 112 gated positive CAMs) |
| mean IoU | 0.3059 |
| IoU ≥ 0.5 hit rate | 0.1429 |
| mAP@0.5 | 0.0077 |

Pipeline counts: 2673 dataset records, 1024 evaluated, 388 ground-truth boxes,
112 CAMs generated, 845 samples skipped by the 0.7 confidence gate, 125 positive cases
with no CAM because of the gate or an empty CAM. CAM boxes are extracted per connected
component (`cam_to_boxes`) from Grad-CAM support thresholded at 0.8; the classifier
never predicts boxes directly.

Pointing-game is markedly lower on this test read (0.4286) than the constant 0.5138
seen across every threshold in the val sweep. Per the project's test-set-tuning
discipline, this is reported as-is; `cam_threshold` is not re-tuned in response, and no
second sweep was run to chase the test number back up.

## Scope of this review

The grid plus three per-sample overlays were inspected directly:
`rsna_00_0174c4bb`, `rsna_05_087bcaa5`, `rsna_12_0c391e0f`. The patterns below are
consistent across those three and with what is visible in the grid; they are not a
census of all 20 overlays, and no per-image scoring was performed. **Provenance note:**
these three files were from the `cam_threshold=0.60`, val-split run; the test-split
regeneration above replaced `results/overlays/rsna/` with a different (test) sample set,
so these exact filenames are no longer in the working tree, only in git history prior to
this session's regeneration. The measured patterns below (component counts, area ratios)
were computed against that val run and have not been re-verified against the new test
overlays — that re-verification is not in scope here.

## Recurring patterns

1. **One blob per image, regardless of how many ground-truth boxes exist.** Grad-CAM
   support is a single contiguous region, so box extraction yields exactly one predicted
   box. In the bilateral cases (`rsna_00`, `rsna_12`) the ground truth is two separate
   tall boxes, one per lung, and the single predicted box spans both of them.
   Measured: at `cam_threshold = 0.6` the thresholded support is a single 8-connected
   component in 170 of 179 images (95%), mean 1.05 components per image.
2. **The predicted box straddles the midline.** In both bilateral cases the extracted
   box covers the mediastinum and spine — the region *between* the two ground-truth
   boxes, which is never part of a lung-opacity annotation. This inflates predicted-box
   area and depresses IoU directly.
3. **Vertical extent is systematically short.** The predicted box consistently stops
   above the lower lung zones where the ground-truth boxes continue down toward the
   costophrenic angles (clearest in `rsna_05` and `rsna_12`). Combined with pattern 2,
   the error is anisotropic: too wide horizontally, too short vertically.
4. **The peak generally lands inside a ground-truth box.** In all three inspected cases
   the maximum-activation region falls within an annotated box, which is consistent with
   a pointing-game score near 0.51 while IoU ≥ 0.5 is reached only 2.75% of the time.

These are properties of turning a diffuse saliency map into an axis-aligned box, and they
reproduce the strict-IoU versus loose-hit-rate divergence reported for CAM-style saliency
in the chest X-ray literature (Saporta et al., *Nature Machine Intelligence* 4:867–878,
2022).

## Why mAP@0.5 is near zero — measured, not inferred

An earlier version of this document attributed the near-zero mAP to pattern 1: one
predicted box cannot match two ground-truth boxes, so most positives were said to be
structurally unmatchable. **That attribution was wrong and the ablation refutes it.**

`cam_to_bbox` returns the extent of *all* thresholded pixels with no connectivity
analysis, so disconnected support silently collapses into one box spanning the gap.
`cam_to_boxes` was added to emit one box per connected component, and the same
evaluation was run both ways (`results/ablation/`, identical checkpoint and samples;
the single-box run reproduces the committed baseline exactly):

| metric | single box | per component | change |
|---|---|---|---|
| mAP@0.5 | 0.00042 | 0.00042 | +0.00000 |
| pointing game | 0.51376 | 0.51376 | +0.00000 |
| IoU ≥ 0.5 hit rate | 0.02752 | 0.02752 | +0.00000 |
| mean IoU | 0.25514 | 0.25700 | +0.00185 |

Nothing moves, because there were no discarded components to recover. The extractor is
still a latent defect — on genuinely disconnected support it merges blobs, and two
disjoint blobs each coinciding with a ground-truth box score IoU 0.333 instead of 1.0
(`test_cam_to_bbox_spans_the_gap_between_disconnected_blobs`) — but it is not what
depresses the metric on this data.

Two measured causes account for it instead:

1. **Boxes are far too large.** Thresholded CAM support covers a median 3.68× the
   annotated box area (mean 5.35×). Best-IoU per positive image has median 0.251 and
   90th percentile 0.420 — the distribution piles up just below the 0.5 criterion.
   37.6% of positives clear IoU ≥ 0.3 while only 2.75% clear 0.5, so the metric is
   dominated by a threshold the boxes systematically miss rather than by gross
   mislocalization.
2. **Two in five emitted boxes sit on negative images.** 70 of the 179 gated CAMs come
   from images with no annotated opacity. mAP pools predictions across images and ranks
   them by score, so these high-confidence false positives occupy top ranks and collapse
   precision — which is why mAP (0.00042) is an order of magnitude worse than the
   per-image hit rate (0.0275) already suggests.

## Threshold sweep: how much is the threshold, how much is the architecture

Cause 1 splits into a size error the threshold controls and a positional error it does
not. Sweeping `cam_threshold` over the **validation** split separates them
(`results/sweep/`, per-component extraction, one CAM per image at every threshold since
CAM generation depends only on the confidence gate). Selection happens on validation;
the chosen value is reported once on the test split.

| threshold | mAP@0.5 | mean IoU | IoU ≥ 0.5 | pointing | pred/GT area | centre distance |
|---|---|---|---|---|---|---|
| 0.30 | 0.00000 | 0.1388 | 0.0000 | 0.5138 | 9.35× | 0.179 |
| 0.40 | 0.00000 | 0.1723 | 0.0000 | 0.5138 | 6.47× | 0.155 |
| 0.50 | 0.00012 | 0.2138 | 0.0183 | 0.5138 | 4.92× | 0.144 |
| 0.60 (default) | 0.00042 | 0.2570 | 0.0275 | 0.5138 | 3.68× | 0.123 |
| 0.70 | 0.00128 | 0.2923 | 0.0550 | 0.5138 | 2.27× | 0.117 |
| **0.80** | **0.00454** | **0.2993** | **0.1193** | 0.5138 | **1.34×** | 0.107 |
| 0.90 | 0.00215 | 0.1968 | 0.0550 | 0.5138 | 0.43× | 0.112 |

Area ratio is the median predicted box area over the mean ground-truth box area on the
same image; centre distance is the median distance from the predicted box centre to the
nearest ground-truth box centre, in units of image width.

**The size error is real and the threshold fixes it.** Area ratio falls from 9.35× to
1.34× and mAP improves 10.8× over the current default; at 0.90 the boxes overshoot to
0.43× and every metric degrades, so 0.80 is an interior optimum rather than the end of a
monotone trend.

**The positional error is not fixed by anything here.** While the area ratio improves
2.7× between 0.60 and 0.80, centre distance moves only from 0.123 to 0.107 and then
plateaus. Even at the optimum 88% of positives still fail the 0.5 IoU criterion.
Pointing-game accuracy is identical at all seven thresholds, which is the internal check
that the sweep changes support geometry only and never the CAM peak.

The residual is the size of the architecture's own quantisation. The final DenseNet121
conv layer is 7×7, so one CAM cell covers roughly 1024/7 ≈ 146 px of a 1024 px image; the
residual median centre error of 0.107 × 1024 ≈ 110 px is about three quarters of one cell.
Once the box is the right size, what is left sits at the resolution limit of the feature
map it was computed from. That is consistent with, though not proof of, a
resolution-bound explanation, and it is the motivation for evaluating multi-scale
features — measured on a scale-free quantity rather than inferred from per-class AUROC
differences, which confound scale with prevalence and label noise.

## Known defect in the artifacts

`results/overlays/rsna/rsna_grid.png` contains **no heatmap**. Mean per-pixel chroma
(max channel − min channel) is 0.82 for the grid against 70.7 for the individual RSNA
overlays and 90.4 for `results/overlays/grid.png`; only 0.5% of grid pixels have chroma
above 30. The grid is being composited from the raw image plus boxes rather than from the
CAM overlay. The per-sample PNGs are correct; only the grid is affected. Not yet fixed.

## What this review does not establish

- Nothing about anatomical correctness beyond gross placement. Whether activation tracks
  actual opacity texture rather than co-occurring image structure was not tested.
- Nothing about the other 13 NIH classes. Only Pneumonia → Lung Opacity was evaluated.
- Nothing about the 845 gated-out samples. A confidence gate at 0.7 means this review
  sees only the cases the classifier was already confident about, which is a selected,
  optimistic subset.
- Grad-CAM at the last DenseNet121 conv layer has a 7×7 spatial resolution upsampled to
  the input size, so fine-grained localization is bounded by the architecture regardless
  of training.

## Open items

- ~~Report the validation-selected `cam_threshold = 0.80` once on the test split.~~
  Done: see "Quantitative anchors" above. Test pointing-game (0.4286) came in below the
  val value (0.5138); the threshold was not revisited in response.
- Fix the grid compositing so `rsna_grid.png` shows the heatmap.
- Compare Grad-CAM++ on the same samples; its higher-order weighting targets the
  multi-instance case, though with 95% single-component support the headroom looks small.
- Evaluate multi-scale features against the residual positional error, which is the one
  failure the threshold cannot reach.

## Note on cause 2, measured

Restricting mAP to images that have at least one ground-truth box raises it from 0.000425
to 0.000613 — a factor of 1.44. Deleting every false-positive box would therefore leave
the metric essentially where it is. The localization failure is geometric, not a
by-product of classifier false positives. The 70-of-179 false-positive rate is still worth
reporting, but as a statement about how permeable the 0.70 confidence gate is, not as an
explanation of the localization numbers.
