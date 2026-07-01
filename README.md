# MEDGUARD-CXR

**Safety disclaimer:** MEDGUARD-CXR is for research and engineering evaluation only. It is not a clinical diagnostic tool, must not be used for patient care, and must not be presented as providing medical advice.

## Current Public Status

Status classification: **D. trained_evaluated_calibrated_grounded (real artifacts present)**.

- **REAL CURRENT STATUS:** NIH classifier training, NIH evaluation, calibration, and RSNA grounding artifacts are present and completed.
- **IDEALIZED FUTURE / NOT YET RUN:** full VLM/QLoRA training, free-text VLM validation, clinical validation, and broader localization milestones remain pending or deferred.
- Real NIH classifier training completed: **yes**.
- NIH evaluation status: **completed**.
- Calibration status: **completed**.
- RSNA grounding status: **completed**, pneumonia/lung-opacity scope only.
- Phase 4A VQA/API/Gradio: **rule-based/fixed-template with real-checkpoint API inference when local artifacts exist; smoke fallback otherwise**.
- Phase 4B VLM/QLoRA: **deferred / not trained / not a current public model milestone**.

The completed model artifact is a DenseNet121 chest X-ray classifier trained on NIH ChestX-ray14 with `make train` in Colab on CUDA. It is not clinically validated. Current RSNA grounding results are limited to NIH `Pneumonia` probability evaluated against RSNA `Lung Opacity`. The binary checkpoint is kept as a local/generated artifact and is ignored by git by default; committed JSON reports preserve the public metric evidence.

## Project Overview

MEDGUARD-CXR is a safety-aware chest X-ray engineering prototype focused on multi-label finding prediction, calibration, abstention, OOD checks, and evidence-gated localization. The project is designed to make uncertainty and provenance visible instead of presenting every output as a confident answer.

## What Is Implemented

- NIH ChestX-ray14 DenseNet121 classifier training/evaluation.
- Temperature-scaling calibration and reliability diagram generation.
- Per-class abstention thresholds.
- RSNA Pneumonia Detection Challenge 2018 grounding validation for `Pneumonia -> Lung Opacity`.
- Grad-CAM overlay generation for confidence-gated RSNA examples.
- Rule-based VQA/API/Gradio scaffolding with safety disclaimers, OOD gates, abstention, and local real-checkpoint classifier inference when `checkpoints/baseline_nih_best.pt` is present.
- Optional Phase 4B VLM/QLoRA adapter-training path, disabled and untrained by default.

## What Has Actually Run

| Component | Command | Artifact | Status | Result | Interpretation |
|---|---|---|---|---|---|
| NIH classifier training | `make train` | `checkpoints/baseline_nih_best.pt`, `results/baseline_nih_train.json` | completed | best epoch 4, validation macro AUROC 0.8246 | Real DenseNet121 checkpoint trained on NIH ChestX-ray14 in Colab CUDA. |
| NIH evaluation | `make eval` | `results/baseline_nih_eval.json` | completed | macro AUROC 0.8037, macro AUPRC 0.2685, macro sensitivity at 90% specificity 0.4708, n=25,596 | Real NIH evaluation artifact. This is not clinical validation. |
| Calibration | `make calibrate` | `results/calibration_report.json`, `results/reliability_diagram.png`, `calibrators/nih_temp_scaling.pkl` | completed | test macro ECE 0.3144 before, 0.3113 after temperature scaling | Calibration report generated on real NIH artifacts; improvement is small. |
| RSNA grounding | `make eval-grounding-rsna` | `results/grounding_rsna_eval.json`, `results/overlays/rsna/rsna_grid.png` | completed | pneumonia AUROC 0.8077, AUPRC 0.5595, pointing-game 0.5138, mean IoU 0.2551, mAP@0.5 0.0004 | Pneumonia-specific RSNA Lung Opacity validation only; no broad localization claim. |
| API health | TestClient `/health` | runtime check | completed | HTTP 200, VLM disabled | Phase 4A app loads; local checkpoint inference is available when artifacts exist, with smoke fallback on missing checkpoints. |
| Gradio build | `build_demo()` smoke check | runtime smoke check | completed | `Blocks` object created | Demo shell builds locally; it is not a validated clinical app. |
| Rule-based VQA | tests and `make eval-vlm-zero-shot` | `results/vlm_zero_shot_eval.json` | completed as rule-based baseline | exact match 0.6091, template adherence 1.0, hallucination rate 0.0 | Rule-based only; no free-text VLM claim. |
| VLM zero-shot | `make eval-vlm-zero-shot` | `results/vlm_zero_shot_eval.json` | blocked for VLM backend | `bitsandbytes` missing; VLM metrics not computed | Experimental path only; rule-based baseline did run. |
| VLM/QLoRA | `make train-vlm` or explicit Colab script command | `results/vlm_lora_train.json` | implemented but not run | epochs completed 0 in current artifact | QLoRA adapter training code exists, but no adapter has been trained yet. |

## What Has Not Been Run Yet

- No QLoRA fine-tuning has been run yet, although the adapter-only training path is now implemented.
- No free-text VLM has been validated; zero-shot/lora backends remain unavailable in the current local environment.
- No prospective or clinical validation has been performed.
- No subgroup performance audit has been completed.
- The Phase 4A API/Gradio path is not a production real-inference service; it is a local research interface with safety gates and real-checkpoint classifier inference when local artifacts are present.

## Artifact Audit

| Artifact | Exists? | Source command | Real / smoke / missing | README? | Report? | Notes |
|---|---:|---|---|---:|---:|---|
| `checkpoints/baseline_nih_best.pt` | local yes / git-ignored | `make train` | real | yes | yes | Epoch-4 DenseNet121 checkpoint from Colab CUDA run; not committed by default. |
| `results/baseline_nih_train.json` | yes | `make train` | real | yes | yes | `mode=nih`, `model_quality_evidence=true`. |
| `results/baseline_nih_eval.json` | yes | `make eval` | real | yes | yes | Macro AUROC/AUPRC reported above. |
| `results/calibration_report.json` | yes | `make calibrate` | real | yes | yes | ECE before/after reported above. |
| `results/reliability_diagram.png` | yes | `make calibrate` | real | yes | yes | Calibration figure artifact. |
| `calibrators/nih_temp_scaling.pkl` | yes | `make calibrate` | real | yes | yes | Saved temperature calibrator. |
| `results/grounding_rsna_eval.json` | yes | `make eval-grounding-rsna` | real | yes | yes | RSNA Pneumonia/Lung Opacity only. |
| `results/overlays/rsna/rsna_grid.png` | yes | `make eval-grounding-rsna` | real engineering artifact | yes | yes | Grad-CAM overlay grid for review, not clinical evidence. |
| API health result | yes | TestClient `/health` | local runtime | yes | yes | VLM disabled; classifier uses local real checkpoint when available and smoke fallback otherwise. |
| Gradio smoke result | yes | `build_demo()` | smoke-tested shell | yes | yes | Builds local UI shell. |
| VQA generator output | local yes / git-ignored | `make vqa-dataset` | generated weak supervision | yes | yes | `data/vqa/synthetic_qa.{train,val,test}.jsonl` exists locally and is ignored by git. |
| `results/vlm_zero_shot_eval.json` | yes | `make eval-vlm-zero-shot` | rule-based baseline plus blocker report | yes | yes | Rule-based metrics computed; VLM backend unavailable because `bitsandbytes` is missing. |
| `results/vlm_lora_train.json` | yes | `make train-vlm` or explicit script command | blocked/deferred | yes | yes | Current artifact has `epochs_completed=0`; QLoRA not trained yet. |

## Scorecard

| Capability | Metric | Phase added | Target | Result |
|---|---|---:|---:|---|
| Multi-label classification | Macro AUROC | P1 | > 0.75 | 0.8037 on NIH evaluation |
| Rare finding detection | Macro AUPRC | P1 | > 0.30 | 0.2685 on NIH evaluation |
| Calibration | ECE lower is better | P2 | < 0.10 target | test macro ECE 0.3144 before, 0.3113 after temperature scaling |
| Selective prediction | Acc @ 20% abstention | P2 | > baseline | not yet summarized as a public score |
| Localization | Pointing game acc | P3 | > 0.50 | 0.5138 on RSNA Lung Opacity confidence-gated examples |
| Cross-dataset localization | NIH Pneumonia -> RSNA Lung Opacity pointing/IoU | P3 | report only | pneumonia AUROC 0.8077, AUPRC 0.5595, mean IoU 0.2551 |
| VQA | Exact match | P4 | > 0.60 | rule-based baseline 0.6091; VLM metrics pending |
| OOD rejection | Cat photo rejected | P4 | pass | smoke-tested in local tests/API shell |

## Data

NIH ChestX-ray14 is used for broad image-level classifier training/evaluation. NIH labels are noisy silver-standard labels mined from reports.

RSNA Pneumonia Detection Challenge 2018 is used only for pneumonia-specific grounding/localization validation. The active mapping is NIH `Pneumonia` to RSNA `Lung Opacity`; all other NIH labels are not mapped to RSNA localization targets.

VinDr-CXR remains deferred future work and is not an active dependency.

## Reproduce

Install and verify:

```bash
make install
make lint
make test
```

Dataset-dependent pipeline:

```bash
make train
make eval
make calibrate
make prepare-rsna
make eval-grounding-rsna
```

API/demo smoke checks:

```bash
python -c "from fastapi.testclient import TestClient; from medguard.api.app import create_app; print(TestClient(create_app()).get('/health').json())"
python -c "from app.gradio_demo import build_demo; print(type(build_demo()).__name__)"
```

## Continuation

The next release-engineering steps are listed in `CONTINUATION_PLAN.md`. Do not claim QLoRA/VLM training, clinical usefulness, prospective validation, or broad multi-abnormality localization before those specific runs and reviews exist.
