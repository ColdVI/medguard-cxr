# MEDGUARD-CXR Model Card

**Safety disclaimer:** This project is for research and engineering evaluation only. It is not a clinical diagnostic tool and must not be used for patient care.

## Model Details

Current Phase 4B status: rule-based VQA remains the safe baseline, with an
optional experimental VLM/QLoRA path added for controlled structured VQA
experiments. The classifier checkpoint available in this workspace is
smoke-trained and is not evidence of diagnostic performance.

## Intended Use

MEDGUARD-CXR is intended for local research and engineering validation of a
safety-aware chest X-ray assistant pipeline. Appropriate uses are checking data
loading, calibration plumbing, abstention behavior, OOD rejection, API response
contracts, and evidence-gated visualization rules. The current smoke-trained
checkpoint must not be used to assess diagnostic quality.

## Out-of-Scope Use

Out-of-scope uses include clinical diagnosis, triage, treatment planning,
patient-care decisions, patient-facing use, public deployment, broad
multi-abnormality localization claims, free-text medical VQA, and any statement
that the current smoke checkpoint is clinically or scientifically validated.

## Training Data

NIH ChestX-ray14 is the intended image-level classification training source.
The current local checkpoint is smoke-mode only.

## Evaluation Data

Localization evaluation uses RSNA Pneumonia Detection Challenge 2018 as the
active dataset, limited to NIH `Pneumonia` mapped to RSNA `Lung Opacity`.
VinDr-CXR is deferred future work only and is not an active dependency.

## Metrics

Phase 3R produces `results/grounding_rsna_eval.json` with RSNA binary
Pneumonia/Lung Opacity classification metrics and weak Grad-CAM localization
metrics. The current report is explicitly marked `WARNING_DO_NOT_USE` because
the available classifier checkpoint is smoke-trained; these values are not
clinical-performance evidence and should not be used as a model card score.

## Calibration

Phase 2 calibration artifacts are available at `results/calibration_report.json`,
`results/reliability_diagram.png`, and `calibrators/nih_temp_scaling.pkl`. The
report is explicitly marked `WARNING_DO_NOT_USE` because it was generated in
synthetic smoke mode after the configured NIH dataset was unavailable.

The smoke calibration report records validation macro ECE 0.322149 before
temperature scaling and 0.321965 after temperature scaling; the test-side smoke
macro ECE is 0.325506 before temperature scaling and 0.325515 after. These
numbers validate report generation and split bookkeeping only, not model
calibration quality.

## Abstention and Uncertainty

The abstention policy is per-class rather than global. The default band from
`configs/calibration.yaml` is `tau_lo=0.30` and `tau_hi=0.70`; Pneumothorax uses
`0.20/0.50`, while Mass and Nodule use `0.20/0.55`. Predictions inside a class's
band abstain and carry explicit abstention metadata.

The smoke selective-risk artifact exercises coverage/risk mechanics and records
the configured thresholds, but it is not evidence that selective prediction
improves a trained model. Phase 4A demo behavior intentionally favors abstention
around uncertain smoke probabilities.

## Explainability

Current explainability is limited to the Phase 3R RSNA path: NIH `Pneumonia`
probability evaluated against RSNA `Lung Opacity` boxes. Grad-CAM is generated
only for non-abstained predictions above the configured confidence gate
(`0.70`). The report records 30 generated Grad-CAM overlays from 1024 evaluated
RSNA validation records, with 7 generated positive CAMs used for pointing-game
accounting.

The current RSNA metrics are `WARNING_DO_NOT_USE`: Pneumonia AUROC 0.423,
pointing-game accuracy 0.000, and mean IoU 0.089. The below-chance AUROC is
consistent with an untrained or smoke-trained checkpoint and is not a
model-quality measurement.

## Ethical Considerations

The project uses public or de-identified datasets only and keeps raw image data
out of source control. NIH labels are silver-standard report-mined labels and
may encode noise or reporting bias. RSNA validation is pneumonia/lung-opacity
specific and must not be generalized to other findings.

A visible research-only disclaimer is required because even smoke outputs can be
misread if shown next to medical images. No current artifact supports a clinical
performance, fairness, or deployment claim.

## Subgroup Performance

No subgroup performance has been validated for the current smoke checkpoint.
Age, sex, scanner/site, and view-position analyses remain future audit work
after a real NIH training run and refreshed evaluation artifacts. Until then,
subgroup behavior is unknown.

## Known Failure Modes

- Smoke-mode outputs are structurally valid but not model-quality evidence.
- The Phase 4A demo defaults to abstention to avoid fabricating confident
  findings from unvalidated weights.

## Caveats and Recommendations

The Phase 4B VLM path is a presentation-layer experiment. It must fall back to
rule-based VQA whenever the image is OOD, the question is unsupported, the
classifier abstains, visual evidence is unavailable for a positive answer, the
VLM mentions unsupported findings, or the VLM conflicts with the classifier.

Before any VLM modeling claim, run a real NIH ChestX-ray14 classifier training
pass, rerun Phase 2 calibration on real validation logits, rerun Phase 3R RSNA
metrics with the smoke warning removed only if appropriate, complete zero-shot
VLM evaluation, and have Claude review the resulting VLM comparison artifact.
