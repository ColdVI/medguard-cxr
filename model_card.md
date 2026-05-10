# MEDGUARD-CXR Model Card

**Safety disclaimer:** This project is for research and engineering evaluation only. It is not a clinical diagnostic tool and must not be used for patient care.

## Model Details

- Model: DenseNet121 chest X-ray classifier.
- Task: multi-label chest X-ray finding prediction on NIH ChestX-ray14 labels.
- Training command: `make train`.
- Checkpoint: `checkpoints/baseline_nih_best.pt` as a local/generated artifact; the binary checkpoint is ignored by git by default.
- Current status: trained, evaluated, calibrated, and run through RSNA pneumonia-specific grounding.
- VQA status: Phase 4A rule-based/fixed-template only.
- VLM/QLoRA status: not trained; deferred optional extension.

## Intended Use

MEDGUARD-CXR is intended for local research and engineering validation of a safety-aware chest X-ray pipeline: data loading, classifier training, evaluation reporting, calibration, abstention behavior, OOD rejection, API contracts, and evidence-gated visualization rules.

## Out-of-Scope Use

Out-of-scope uses include clinical diagnosis, triage, treatment planning, patient-care decisions, patient-facing use, public deployment as a medical system, broad multi-abnormality localization claims, and any claim that the project has clinical utility.

## Training Data

The classifier was trained on NIH ChestX-ray14 using the repository `configs/baseline_nih.yaml` path assumptions. NIH labels are noisy silver-standard labels mined from reports. The current training artifact reports `mode=nih`, CUDA execution, best epoch 4, and validation macro AUROC 0.8246.

## Evaluation Data

NIH ChestX-ray14 evaluation was run with `make eval`; `results/baseline_nih_eval.json` reports macro AUROC 0.8037, macro AUPRC 0.2685, and macro sensitivity at 90% specificity 0.4708 over 25,596 samples.

RSNA Pneumonia Detection Challenge 2018 was used for the single approved grounding bridge: NIH `Pneumonia` probability evaluated against RSNA `Lung Opacity`. No other NIH finding has RSNA localization support in this project.

VinDr-CXR remains deferred future work.

## Calibration

Calibration was run with `make calibrate`.

- Validation macro ECE before temperature scaling: 0.2388.
- Validation macro ECE after temperature scaling: 0.2369.
- Test macro ECE before temperature scaling: 0.3144.
- Test macro ECE after temperature scaling: 0.3113.
- Reliability diagram: `results/reliability_diagram.png`.
- Calibrator: `calibrators/nih_temp_scaling.pkl`.

Calibration improved ECE slightly but remains imperfect. The model should still expose uncertainty and abstention metadata.

## Abstention and Uncertainty

The abstention policy is per-class rather than global. The default band from `configs/calibration.yaml` is `tau_lo=0.30` and `tau_hi=0.70`; Pneumothorax uses `0.20/0.50`, while Mass and Nodule use `0.20/0.55`. Predictions inside a class's band abstain and carry explicit abstention metadata.

## Explainability and Grounding

Current grounding is limited to RSNA Pneumonia Detection Challenge 2018:

- NIH class: `Pneumonia`.
- RSNA target: `Lung Opacity`.
- Evaluated records: 1024.
- Generated Grad-CAM overlays: 179.
- Pneumonia AUROC: 0.8077.
- Pneumonia AUPRC: 0.5595.
- Pointing-game accuracy: 0.5138.
- Mean IoU: 0.2551.
- mAP@0.5: 0.0004.

These are research evaluation metrics for a pneumonia-specific cross-dataset check. Grad-CAM overlays are weak engineering artifacts and are not clinical evidence.

## VQA, API, and Demo Status

Phase 4A is rule-based/fixed-template. It keeps safety disclaimers visible and uses abstention/OOD checks. The default API/Gradio shell remains smoke-tested and template-driven; it should not be described as a real clinical assistant.

Phase 4B VLM/QLoRA scaffolding exists but is not trained. Current VLM reports show blocked/deferred status and `epochs_completed=0`.

## Ethical Considerations

The project uses public or de-identified datasets only and keeps raw image data out of source control. NIH labels can encode report-mining noise, acquisition bias, subgroup imbalance, and site/view differences. RSNA validation is pneumonia/lung-opacity specific and must not be generalized to other findings.

## Subgroup Performance

No subgroup performance audit has been completed. Age, sex, scanner/site, and view-position analyses remain future work.

## Known Limitations

- No clinical validation or prospective evaluation.
- NIH labels are noisy silver-standard labels.
- Calibration improved only slightly and remains imperfect.
- RSNA grounding supports only pneumonia/lung-opacity evaluation.
- Grad-CAM boxes are weak localization proxies.
- Rule-based VQA is not free-text VLM reasoning.
- QLoRA/VLM has not been trained.

## Recommendations

Before any stronger public claim, rerun the full pipeline from data preparation through evaluation in a clean environment, complete subgroup/error analysis, and have the real metric artifacts reviewed. Do not claim clinical utility, deployment readiness, or QLoRA/VLM training from the current artifacts.
