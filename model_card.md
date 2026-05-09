# MEDGUARD-CXR Model Card

**Safety disclaimer:** This project is for research and engineering evaluation only. It is not a clinical diagnostic tool and must not be used for patient care.

## Model Details

Current Phase 4A status: rule-based VQA and safety/API scaffolding only. The
classifier checkpoint available in this workspace is smoke-trained and is not
evidence of diagnostic performance.

## Intended Use

TBD.

## Out-of-Scope Use

TBD.

## Training Data

NIH ChestX-ray14 is the intended image-level classification training source.
The current local checkpoint is smoke-mode only.

## Evaluation Data

Localization evaluation uses RSNA Pneumonia Detection Challenge 2018 as the
active dataset, limited to NIH `Pneumonia` mapped to RSNA `Lung Opacity`.
VinDr-CXR is deferred future work only and is not an active dependency.

## Metrics

TBD.

## Calibration

TBD.

## Abstention and Uncertainty

TBD.

## Explainability

TBD.

## Ethical Considerations

TBD.

## Subgroup Performance

TBD.

## Known Failure Modes

- Smoke-mode outputs are structurally valid but not model-quality evidence.
- The Phase 4A demo defaults to abstention to avoid fabricating confident
  findings from unvalidated weights.

## Caveats and Recommendations

TBD.
