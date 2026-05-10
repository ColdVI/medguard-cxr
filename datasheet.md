# MEDGUARD-CXR Datasheet

**Safety disclaimer:** This project uses public or de-identified data only. It is not intended for clinical deployment or patient-care decisions.

## Motivation

The active data setup supports research and engineering validation of a safety-aware chest X-ray pipeline: NIH classifier training/evaluation, calibration and abstention reports, RSNA pneumonia/lung-opacity grounding, and Phase 4A rule-based safety/API workflows.

## Dataset Roles

- NIH ChestX-ray14: broad image-level multi-label classifier training and evaluation.
- RSNA Pneumonia Detection Challenge 2018: pneumonia-specific grounding/localization check for NIH `Pneumonia -> RSNA Lung Opacity`.
- Synthetic VQA JSONL: optional future generated records from structured labels, probabilities, and templates; current VQA JSONL files are not present.
- VinDr-CXR: deferred future work due to access friction; not an active dependency.

## Composition

NIH ChestX-ray14 provides image-level labels for 14 thoracic findings. Labels are silver-standard labels mined from reports and can be noisy.

RSNA Pneumonia Detection Challenge 2018 provides public training labels for `Lung Opacity`, including bounding boxes for positive opacity cases. Hidden Kaggle test labels are not used.

## Collection Process

NIH and RSNA are external public or de-identified datasets obtained by the operator under their source terms. Raw medical images and provider annotations are not redistributed in this repository.

## Preprocessing and Splits

NIH training/evaluation uses `configs/baseline_nih.yaml`, with official list files preferred when present and patient-disjoint split behavior enforced by the dataset loader.

RSNA bounding boxes are parsed from `x, y, width, height` pixel coordinates and converted to normalized `xyxy` boxes in `[0, 1]`. Public RSNA training labels are split deterministically by `patientId`; hidden challenge test labels are not used.

## What The Data Supports

- NIH supports image-level multi-label classifier training/evaluation.
- RSNA supports pneumonia/lung-opacity binary classification and weak Grad-CAM localization evaluation for that target only.
- Current data does not support broad multi-abnormality localization claims.
- Current data does not establish clinical utility or prospective safety.

## Distribution

The repository may contain configs, scripts, tests, JSON summaries, calibration files, and review PNGs. Raw datasets should remain under ignored runtime paths such as `data/nih/` and `data/rsna/`.

## Maintenance

Any future real training run must regenerate:

- `checkpoints/baseline_nih_best.pt`
- `results/baseline_nih_train.json`
- `results/baseline_nih_eval.json`
- `results/calibration_report.json`
- `results/reliability_diagram.png`
- `calibrators/nih_temp_scaling.pkl`
- `results/grounding_rsna_eval.json`
- RSNA overlay samples
- README/model-card/report metric tables

## Ethical and Safety Notes

The datasets can carry label noise, acquisition bias, subgroup imbalance, and view-position differences. Documentation and UI surfaces must keep research-only language visible. No artifact in this repository establishes clinical validation, patient-care readiness, or fairness across subgroups.
