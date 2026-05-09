# MEDGUARD-CXR

**Safety disclaimer:** MEDGUARD-CXR is for research and engineering evaluation only. It is not a clinical diagnostic tool, must not be used for patient care, and must not be presented as providing medical advice.

## Project Overview

TBD.

## Intended Use

TBD.

## Out-of-Scope Use

TBD.

## Safety Principles

- The system may abstain instead of returning a prediction.
- Displayed positive findings must include visual evidence in the demo/UI when implemented.
- Public, de-identified datasets only.

## Scorecard

| Capability | Metric | Phase added | Target | Result |
|------------|--------|------------|--------|--------|
| Multi-label classification | Macro AUROC | P1 | > 0.75 | TBD |
| Rare finding detection | Macro AUPRC | P1 | > 0.30 | TBD |
| Calibration | ECE ↓ | P2 | < 0.10 | TBD |
| Selective prediction | Acc @ 20% abstention | P2 | > baseline | TBD |
| Localization | Pointing game acc | P3 | > 0.50 | TBD |
| Cross-dataset localization | NIH Pneumonia→RSNA Lung Opacity pointing/IoU | P3 | report only | TBD (smoke) |
| VQA | Exact match | P4 | > 0.60 | TBD |
| OOD rejection | Cat photo rejected | P4 | pass | TBD |

## Repository Layout

TBD.

## Setup

```bash
make install
make test
```

## Data Preparation

The active localization dataset is RSNA Pneumonia Detection Challenge 2018.
Only NIH `Pneumonia` maps to RSNA `Lung Opacity`; all other NIH labels are not
applicable for RSNA localization. VinDr-CXR is deferred future work only and is
not an active pipeline dependency.

## Training

TBD.

## Evaluation

Phase 3R adds RSNA Pneumonia Detection Challenge 2018 validation for the single
approved cross-dataset localization bridge: NIH `Pneumonia` probability against
RSNA `Lung Opacity` labels and boxes. The current generated report is
`results/grounding_rsna_eval.json`.

Current Phase 3R report scope: 1024 validation records from the deterministic
RSNA split, including 237 positive and 787 negative cases. The report is marked
`WARNING_DO_NOT_USE` because the available NIH classifier checkpoint is
smoke-trained; the numbers validate the plumbing and data alignment only, not
clinical or model quality.

## Calibration and Abstention

TBD.

## Localization

TBD.

## Demo

Phase 4A ships a local rule-based VQA/API/Gradio demo. It runs in smoke mode by
default and must not be interpreted as real model performance.

## Reproducibility

TBD.

## Limitations

TBD.
