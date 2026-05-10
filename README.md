# MEDGUARD-CXR

**Safety disclaimer:** MEDGUARD-CXR is for research and engineering evaluation only. It is not a clinical diagnostic tool, must not be used for patient care, and must not be presented as providing medical advice.

## Project Overview

MEDGUARD-CXR is a research-grade chest X-ray engineering project focused on
safety plumbing: multi-label finding prediction, calibration, abstention,
OOD rejection, and evidence-gated localization checks. The current workspace
contains smoke-trained model artifacts and rule-based Phase 4A scaffolding; it
does not contain a clinically validated or real-performance checkpoint.

## Intended Use

Intended use is local research and engineering validation of the MEDGUARD-CXR
safety contract: data loading, metric generation, calibration reports,
abstention behavior, OOD handling, API contracts, and evidence display rules.
Outputs are for debugging the pipeline only.

## Out-of-Scope Use

Out-of-scope uses include clinical diagnosis, triage, treatment planning,
patient-care decisions, public deployment, broad abnormality localization
claims, free-text medical VQA, and any claim that the current smoke checkpoint
measures real model quality.

## Safety Principles

- The system may abstain instead of returning a prediction.
- Displayed positive findings must include visual evidence in the demo/UI when implemented.
- Public, de-identified datasets only.

## Scorecard

| Capability | Metric | Phase added | Target | Result |
|------------|--------|------------|--------|--------|
| Multi-label classification | Macro AUROC | P1 | > 0.75 | Not real-evaluated; smoke report has null AUROC and is `WARNING_DO_NOT_USE` |
| Rare finding detection | Macro AUPRC | P1 | > 0.30 | Not real-evaluated; smoke report has null AUPRC and is `WARNING_DO_NOT_USE` |
| Calibration | ECE ↓ | P2 | < 0.10 | Smoke validation macro ECE 0.322 before, 0.322 after temperature scaling; `WARNING_DO_NOT_USE` |
| Selective prediction | Acc @ 20% abstention | P2 | > baseline | Not real-evaluated; smoke selective-risk mechanics only, `WARNING_DO_NOT_USE` |
| Localization | Pointing game acc | P3 | > 0.50 | 0.000 on RSNA Lung Opacity positives with generated CAMs; `WARNING_DO_NOT_USE` smoke checkpoint |
| Cross-dataset localization | NIH Pneumonia→RSNA Lung Opacity pointing/IoU | P3 | report only | Pneumonia AUROC 0.423, pointing 0.000, mean IoU 0.089; `WARNING_DO_NOT_USE` smoke checkpoint |
| VQA | Exact match | P4 | > 0.60 | Rule-based baseline available; Phase 4B VLM evaluation path is experimental and has no real metrics yet |
| OOD rejection | Cat photo rejected | P4 | pass | Phase 4A smoke/API test coverage only; no clinical-performance score |

## Repository Layout

Core implementation lives under `src/medguard/`, command-line workflows under
`scripts/`, configuration under `configs/`, and generated summary artifacts
under `results/`. Raw datasets are not committed; local data roots are runtime
inputs only.

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

The current available NIH classifier checkpoint is smoke-trained because the
configured NIH dataset was not available for a full run in this workspace. A
real NIH ChestX-ray14 training pass is required before downstream metrics can be
used as model-quality evidence.

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

The below-chance RSNA Pneumonia AUROC of 0.423 is consistent with an untrained
or smoke-trained checkpoint and is not a model-quality measurement.

## Calibration and Abstention

Phase 2 calibration artifacts are present in `results/calibration_report.json`
and `results/reliability_diagram.png`, but they are also marked
`WARNING_DO_NOT_USE` because they were produced in synthetic smoke mode. The
abstention policy uses per-class bands from `configs/calibration.yaml`
(`tau_lo=0.30`, `tau_hi=0.70` by default, with lower positive thresholds for
Pneumothorax, Mass, and Nodule).

## Localization

Active localization validation is limited to NIH `Pneumonia` mapped to RSNA
`Lung Opacity`. Grad-CAM overlays are generated only when confidence clears the
configured gate and the prediction is not abstained; they are weak visual
engineering artifacts, not clinical evidence.

## Demo

Phase 4A ships a local rule-based VQA/API/Gradio demo. It runs in smoke mode by
default and must not be interpreted as real model performance.

## Experimental Phase 4B VLM Path

Phase 4B adds an optional Qwen2.5-VL/QLoRA path for controlled, structured VQA
experiments. The rule-based VQA path remains the safe baseline and fallback. The
VLM is never the primary decision-maker: classifier probabilities, abstention,
OOD gates, evidence availability, hallucination checks, and the safety
disclaimer are enforced before any VLM answer can be displayed.

The local default keeps VLM loading and QLoRA training disabled unless
configured resources are available. Current VLM outputs and training reports are
`WARNING_DO_NOT_USE` engineering artifacts until a real NIH checkpoint,
refreshed calibration, and reviewed VLM evaluation results exist.

## Reproducibility

Configs record fixed seeds and runtime paths. Phase 3R metrics are written to
`results/grounding_rsna_eval.json`, and design decisions are logged in
`DECISIONS.md`. Hidden RSNA challenge test labels are not used.

## Limitations

The current checkpoint is smoke-trained, Phase 3R AUROC is below chance, and
Phase 4B VLM/QLoRA is experimental, optional, and disabled by default. A real
NIH checkpoint, refreshed calibration, rerun RSNA metrics, and reviewed VLM
evaluation are required before any model-quality claim.
