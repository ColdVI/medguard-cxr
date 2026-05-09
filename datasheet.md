# MEDGUARD-CXR Datasheet

**Safety disclaimer:** This project uses public or de-identified data only. It is not intended for clinical deployment or patient-care decisions.

## Motivation

TBD.

## Composition

- NIH ChestX-ray14: planned image-level multi-label classification data.
- RSNA Pneumonia Detection Challenge 2018: active localization audit data for
  the pneumonia/lung-opacity path only.
- Synthetic VQA JSONL: generated from model outputs and fixed safety templates.
- VinDr-CXR: deferred future work only, not an active dependency.

## Collection Process

TBD.

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

TBD.

## Maintenance

TBD.

## Ethical and Safety Notes

TBD.
