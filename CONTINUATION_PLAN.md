# MEDGUARD-CXR Continuation Plan

**Safety disclaimer:** MEDGUARD-CXR is a research and engineering prototype only. Do not use it for patient care.

## 1. Current State

Status classification: **D. trained_evaluated_calibrated_grounded**.

The canonical local artifact paths now contain the real Colab run outputs:

- NIH classifier training completed with `make train`.
- NIH evaluation completed with `make eval`.
- Calibration completed with `make calibrate`.
- RSNA pneumonia-specific grounding completed with `make eval-grounding-rsna`.
- Phase 4A API/Gradio/VQA remains rule-based and smoke-tested.
- Phase 4B VLM/QLoRA adapter training is implemented, disabled by default, and not trained in current artifacts.

## 2. Completed Commands

```bash
make train
make eval
make calibrate
make eval-grounding-rsna
```

The reported Colab training evidence was: `Training completed in NIH mode. Best epoch: 4`.

## 3. Existing Artifacts

| Artifact | Status | Interpretation |
|---|---|---|
| `checkpoints/baseline_nih_best.pt` | real local artifact, git-ignored | DenseNet121 NIH checkpoint from epoch 4. |
| `results/baseline_nih_train.json` | real | Training report, `mode=nih`, CUDA run. |
| `results/baseline_nih_eval.json` | real | NIH evaluation metrics. |
| `results/calibration_report.json` | real | Temperature-scaling calibration report. |
| `results/reliability_diagram.png` | real | Calibration figure. |
| `calibrators/nih_temp_scaling.pkl` | real | Saved calibrator. |
| `results/grounding_rsna_eval.json` | real | RSNA Pneumonia/Lung Opacity evaluation. |
| `results/overlays/rsna/rsna_grid.png` | real engineering artifact | RSNA Grad-CAM grid for review. |
| `results/vlm_zero_shot_eval.json` | blocked | Missing VQA test JSONL. |
| `results/vlm_lora_train.json` | deferred/currently untrained | QLoRA adapter training code exists; current artifact has epochs completed 0. |

## 4. Commands To Run Next

Run these only when you want to reproduce or refresh the release artifacts.

```bash
make lint
make test
python -c "import medguard; print('medguard import ok')"
```

If NIH and RSNA data are available in the configured paths:

```bash
make train
make eval
make calibrate
make prepare-rsna
make eval-grounding-rsna
```

Then package outputs:

```bash
zip -r medguard_release_artifacts.zip results calibrators checkpoints README.md model_card.md datasheet.md DECISIONS.md CONTINUATION_PLAN.md report
```

## 5. Expected Output Of Each Command

- `make train`: updates `checkpoints/baseline_nih_best.pt` and `results/baseline_nih_train.json`.
- `make eval`: updates `results/baseline_nih_eval.json` with NIH macro AUROC/AUPRC and sensitivity at 90% specificity.
- `make calibrate`: updates `calibrators/nih_temp_scaling.pkl`, `calibrators/nih_temp_scaling.json`, `results/calibration_report.json`, and `results/reliability_diagram.png`.
- `make prepare-rsna`: updates `results/rsna_manifest.csv`.
- `make eval-grounding-rsna`: updates `results/grounding_rsna_eval.json` and `results/overlays/rsna/`.

## 6. How To Interpret Outputs

- NIH metrics are classifier evaluation metrics on public NIH labels, not clinical validation.
- Calibration ECE reports probability calibration quality; lower is better, but a small improvement does not make the model clinically safe.
- RSNA metrics are pneumonia/lung-opacity only.
- Grad-CAM overlays are weak review artifacts, not clinical evidence.
- API/Gradio smoke checks confirm app wiring and safety disclaimers, not real assistant quality.

## 7. Updates After Each Run

After any rerun:

1. Read each updated JSON artifact.
2. Confirm whether `WARNING_DO_NOT_USE` is present.
3. Update README scorecard values only from real artifacts.
4. Update `model_card.md` calibration/evaluation sections.
5. Update `report/sections/06_results.tex`.
6. Append `DECISIONS.md` with the command, environment, and artifact status.

## 8. What Not To Claim

Do not claim:

- clinical usefulness
- diagnostic safety
- prospective validation
- broad multi-abnormality localization
- QLoRA/VLM training
- free-text medical chatbot behavior
- subgroup fairness
- deployment readiness as a medical system

Do not claim QLoRA/VLM was trained until `results/vlm_lora_train.json` records real completed epochs and reviewed evaluation metrics.

## 9. Optional Phase 4B Gate

Phase 4B may be reactivated only after the owner explicitly chooses to do so. The minimum Colab command order is:

```bash
pip install -e ".[dev,vlm]"
make vqa-dataset
make eval-vlm-zero-shot
python3 scripts/train_vlm_lora.py --config configs/vlm_lora.yaml --enable-training --max-steps 200 --limit-train 512 --limit-val 128
make eval-vlm-lora
make eval-vlm-compare
```

Remove the `--max-steps` and limit flags only when you intentionally want a longer run. Before the training/evaluation commands complete successfully, the public status remains: **VLM/QLoRA implemented, not trained**.
