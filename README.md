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
| Cross-dataset | Macro AUROC NIH→VinDr | P3 | report only | TBD |
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

TBD.

## Training

TBD.

## Evaluation

TBD.

## Calibration and Abstention

TBD.

## Localization

TBD.

## Demo

TBD.

## Reproducibility

TBD.

## Limitations

TBD.
