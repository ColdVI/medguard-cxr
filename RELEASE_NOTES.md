# MEDGUARD-CXR Release Notes

## Current Results Publication

Release scope: research and engineering portfolio artifact. Not for clinical use.

## Completed

- Real NIH ChestX-ray14 DenseNet121 classifier training completed with `make train`.
- Best epoch reported by the training artifact: 4.
- Checkpoint written to `checkpoints/baseline_nih_best.pt`.
- NIH evaluation completed with `make eval`.
- Calibration completed with `make calibrate`.
- RSNA Pneumonia Detection Challenge grounding completed with `make eval-grounding-rsna`.
- Report LaTeX source added under `report/`.

## Smoke-Tested

- Phase 4A API health route via FastAPI TestClient.
- Phase 4A Gradio build path.
- Rule-based/fixed-template VQA safety behavior.
- OOD and disclaimer/provenance middleware through tests.

## Trained But Not Fully Extended

- The trained model is the DenseNet121 NIH classifier.
- The Phase 4A API/demo shell is not a production real-inference service.
- The project has not completed subgroup analysis or prospective validation.

## Pending Future Run

- Optional VQA JSONL generation.
- VLM zero-shot evaluation.
- QLoRA fine-tuning run and evaluation; adapter-only training code is available but no adapter has been trained in the current artifacts.
- VLM/QLoRA comparison review.
- Subgroup and error analysis.

## Explicit Non-Claims

- No clinical validation.
- No patient-care use.
- No broad multi-abnormality localization claim.
- No QLoRA/VLM training claim.
- No free-text medical chatbot claim.
