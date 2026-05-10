# MEDGUARD-CXR Datasheet

**Safety disclaimer:** This project uses public or de-identified data only. It is not intended for clinical deployment or patient-care decisions.

## Motivation

The active data setup supports research and engineering validation of the
MEDGUARD-CXR safety pipeline: NIH image-level classification scaffolding, Phase
2 calibration and abstention reports, Phase 3R RSNA pneumonia/lung-opacity
localization validation, and Phase 4A rule-based safety/API workflows. It is not
assembled for clinical deployment or patient-care decision making.

## Composition

- NIH ChestX-ray14: planned image-level multi-label classification data.
- RSNA Pneumonia Detection Challenge 2018: active localization audit data for
  the pneumonia/lung-opacity path only.
- Synthetic VQA JSONL: generated from model outputs and fixed safety templates.
- VinDr-CXR: deferred future work only, not an active dependency.

## Collection Process

NIH ChestX-ray14 and RSNA Pneumonia Detection Challenge 2018 are external public
or de-identified datasets obtained by the operator under their source terms.
NIH image labels are silver-standard labels mined from reports. RSNA public
training labels provide `Lung Opacity` image-level targets and bounding boxes;
hidden Kaggle challenge test labels are not used. Synthetic VQA records are
derived from structured labels, probabilities, abstention decisions, and fixed
safety templates; they are not patient-authored reports.

## Preprocessing

RSNA bounding boxes are parsed from `x, y, width, height` pixel coordinates and
stored as normalized `xyxy` boxes in `[0, 1]`. Public RSNA train labels are split
deterministically by `patientId`; hidden challenge test labels are not used.
Negative RSNA rows are retained for Phase 3R binary Pneumonia/Lung Opacity
classification metrics and carry no localization boxes.

## Recommended Uses

Research and engineering evaluation of safety scaffolding, abstention, OOD
gates, calibration plumbing, and user-interface guardrails.

## Out-of-Scope Uses

Clinical decision support, diagnosis, treatment recommendation, public network
deployment, or claims about real-world model performance.

## Distribution

Raw medical images and provider annotations are not redistributed in this
repository. The repo may contain configs, scripts, tests, synthetic smoke
artifacts, and JSON/PNG summary outputs. Operators must download external
datasets from their official providers and keep local data under ignored runtime
paths such as `data/rsna/`.

## Maintenance

Dataset paths, split parameters, and metric outputs are maintained through YAML
configs and `DECISIONS.md`. Any real training run must regenerate calibration,
RSNA Phase 3R metrics, overlay samples, and documentation labels so smoke-mode
warnings are not mixed with real-checkpoint results.

## Ethical and Safety Notes

The datasets can carry label noise, acquisition bias, subgroup imbalance, and
view-position differences. Current artifacts do not establish subgroup
performance or clinical safety. Documentation and UI surfaces must keep
`WARNING_DO_NOT_USE` and research-only language visible whenever smoke-trained
or synthetic-smoke outputs are shown.
