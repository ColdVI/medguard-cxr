# MEDGUARD-CXR — MASTER ORCHESTRATION PROMPT v2.0
> Single source of truth. Read fully before acting.
> Project Owner : Anıl Keskin
> Produced by   : Claude Opus 4.7 (architecture + safety)
>                 GPT-5.5 Codex (execution review)
>                 Gemini 3 Flash (audit + conflict resolution)
> Version history: v1.0 Claude draft → v1.1 GPT review → v2.0 merged

---

## 0. WHAT THIS PROJECT IS

> A research-grade, safety-aware chest X-ray assistant that detects thoracic
> findings, localizes suspicious regions, estimates uncertainty, and **refuses**
> low-confidence cases instead of hallucinating.

### What this is NOT
- Not a clinical diagnostic tool.
- Not a "I trained a classifier" demo.
- Not a functional repo without evaluation metrics.

---

## 1. HARD CONSTRAINTS (ALL MODELS, ALL PHASES, NON-NEGOTIABLE)

| ID | Constraint | Scope |
|----|-----------|-------|
| C1 | Model may abstain. Not every input gets a prediction. | All phases |
| C2 | No clinical claim in code, docs, API responses, or UI. | All phases |
| C3 | Every prediction must expose a confidence/uncertainty value. | v0.1+ |
| C4 | Every **displayed** positive prediction must have visual evidence (CAM/bbox). | Demo/UI only in v0.1; all outputs in v0.3 |
| C5 | Evaluation must include calibration metrics, not only AUROC. | v0.2+ |
| C6 | At least one cross-dataset test (train NIH → eval VinDr or vice versa). | v0.2+ |
| C7 | Reproducibility: fixed seeds + config YAML minimum. MLflow/W&B added in v0.2. | v0.1+ |
| C8 | No patient data. Public / de-identified datasets only. | All phases |

*Note on C4: In v0.1 baseline, the API may return predictions without CAM.
The Gradio UI must always show visual evidence for positive predictions.
By v0.3, all API positive predictions must include evidence.*

---

## 2. MODEL OWNERSHIP TABLE

Every file in the repo has a primary owner. Owners write first; others review.
No model silently rewrites another model's primary file.

| File / Area | Primary Owner | Reviewer |
|-------------|--------------|---------|
| Repo skeleton, Makefile, Docker, CI | Codex | Claude |
| `data/nih.py`, `data/vindr.py`, `data/dicom.py`, `data/transforms.py` | Codex | Claude |
| `models/classifier.py` (architecture, loss) | Codex | Claude |
| `models/calibration.py` (temperature, isotonic) | **Claude** | Codex |
| `models/vlm.py` (QLoRA wiring, training loop) | Codex | Claude |
| `safety/abstention.py` (policy logic, thresholds) | **Claude** | Codex |
| `safety/ood.py` | Codex | Claude |
| `eval/classification_metrics.py` | Codex | Claude |
| `eval/calibration_metrics.py` (ECE, reliability diagram) | **Claude** | Codex |
| `eval/localization_metrics.py` | Codex | Claude |
| `explain/gradcam.py` | Codex | Claude |
| `api/app.py`, `api/schemas.py` | Codex | Claude |
| `app/gradio_demo.py` | Codex | Claude |
| `notebooks/01_dataset_audit.ipynb` | **Gemini** | Claude |
| `notebooks/03_error_analysis.ipynb` (Grad-CAM visual audit) | **Gemini** | Claude |
| `model_card.md`, `datasheet.md` | **Claude** | Gemini |
| `README.md` | **Claude** | Codex |
| `DECISIONS.md` | All (append-only) | Owner |

---

## 3. HANDSHAKE PROTOCOL (MANDATORY — ALL MODELS — EVERY SESSION)

### 3a. Session opening block
Paste this at the top of every response when starting or continuing a phase:

```
╔══════════════════════════════════════════════════════════════╗
║  MEDGUARD HANDSHAKE                                          ║
║  Model      : <name and version>                             ║
║  Phase      : <phase ID and name>                            ║
║  Status     : starting | continuing | reviewing              ║
║  Prior work : accepted | needs_revision — <reason if any>    ║
║  Planned    : <list of files you intend to touch this turn>  ║
║  Handoff to : <next model or "owner for approval">           ║
╚══════════════════════════════════════════════════════════════╝
```

### 3b. Session closing block
Paste this at the end of every response when finishing or pausing:

```
╔══════════════════════════════════════════════════════════════╗
║  MEDGUARD HANDOFF NOTE                                       ║
║  Files changed  : <list>                                     ║
║  Commands run   : <make targets or scripts>                  ║
║  Results        : <metrics, errors, or "smoke test passed">  ║
║  Known issues   : <anything incomplete or risky>             ║
║  Blocked by     : <missing data, credential, owner decision> ║
║  Next model     : <recommended model for next step>          ║
║  Next action    : <first thing next model should do>         ║
╚══════════════════════════════════════════════════════════════╝
```

### 3c. Review block (for reviewer role)
When acting as reviewer on another model's output:

```
╔══════════════════════════════════════════════════════════════╗
║  MEDGUARD REVIEW                                             ║
║  Reviewer   : <model name>                                   ║
║  Reviewing  : <phase and files>                              ║
║  CRITICAL   : <blockers — must fix before handoff>           ║
║  MAJOR      : <important issues — fix in this phase>         ║
║  MINOR      : <nice to fix but not blocking>                 ║
║  SAFETY     : <any constraint violation from Section 1>      ║
║  VERDICT    : approved | approved_with_conditions | rejected  ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 4. PHASES

---

### PHASE 0A — Repo Skeleton + Environment
**Owner: GPT-5.5 Codex**
**Trigger: Owner says "start Phase 0A".**
**Entry condition: None. This is the first phase.**

Deliverables:
- [ ] Full repo skeleton matching Section 6 exactly (empty files with docstrings)
- [ ] `pyproject.toml` with all dependencies pinned
- [ ] `configs/baseline_nih.yaml` with all hyperparameters (no hardcoding)
- [ ] `Makefile` targets: `install`, `prepare-nih`, `prepare-vindr`, `train`, `eval`, `calibrate`, `test`, `demo`, `lint`
- [ ] `Dockerfile` with CPU and GPU targets
- [ ] `.github/workflows/ci.yml`: import smoke test on CPU only
- [ ] `DECISIONS.md` initialized (empty log)
- [ ] `README.md` shell with section headers (content TBD)

Exit condition (REQUIRED — blocks handoff):
- `make install` completes without error in a clean venv
- All Python files importable (`python -c "import medguard"`)

Exit condition (TARGET — not a blocker):
- CI passes on push

Handoff: → Phase 0B (Gemini) in parallel with → Phase 1 (Codex)

---

### PHASE 0B — Data Audit
**Owner: Gemini 3 Flash**
**Trigger: Can start in parallel with Phase 1. Does not block Phase 1.**
**Entry condition: Owner provides dataset documentation links.**

Deliverables:
- [ ] Data leakage risk report (patient overlap between official splits?)
- [ ] Label quality note: NIH labels are silver-standard (NLP-mined from reports).
      Document which of the 14 labels are most noisy.
- [ ] Class imbalance table (per-label prevalence in train set)
- [ ] Subgroup audit plan: age, sex, view position (PA vs AP vs LL)
- [ ] pos_weight recommendation per class for weighted BCE loss
- [ ] Recommended train/val/test split rationale

Exit condition: Written report in `notebooks/01_dataset_audit.ipynb`.
Handoff: → Claude for review → Owner for approval.

---

### PHASE 1 — NIH ChestX-ray14 Baseline
**Owner: GPT-5.5 Codex**
**Entry condition: Phase 0A exit conditions met.**

Deliverables:
- [ ] `src/medguard/data/nih.py`
  - Multi-label binary target vector (14 classes)
  - Respects official train/val/test patient split from `train_val_list.txt` / `test_list.txt`
  - MONAI transforms: resize 224, random HFlip (train only), then normalization as follows:
    - If using ImageNet-pretrained backbone (DenseNet121 default): convert grayscale to 3-channel repeat, apply ImageNet mean/std normalization
    - If using TorchXRayVision pretrained backbone: use that model's expected normalization (do NOT apply ImageNet stats)
    - Document which normalization is active in `configs/baseline_nih.yaml` under `preprocessing.normalization`
  - Returns `{image: Tensor[1,224,224], label: Tensor[14], patient_id: str, path: str}`
  - DataLoader settings must come from config YAML, not hardcoded:
    - GPU training default: `num_workers=4`, `pin_memory=True`, `prefetch_factor=2`
    - CI / CPU smoke test: `num_workers=0`, `pin_memory=False`
    - Mac / MPS: `num_workers=0` or `num_workers=2`, `pin_memory=False`
- [ ] `src/medguard/data/transforms.py`: shared transform factory
- [ ] `src/medguard/models/classifier.py`
  - DenseNet121 backbone (ImageNet pretrained)
  - Multi-label head: `nn.Linear` outputting **raw logits** during training
  - Use `BCEWithLogitsLoss(pos_weight=...)` — sigmoid is applied inside the loss, do NOT apply sigmoid before the loss
  - Apply `torch.sigmoid()` only during inference/evaluation to convert logits to probabilities
  - NOT Softmax — diseases are not mutually exclusive
  - `pos_weight` per class from Phase 0B audit (or computed inline from training set prevalence if audit pending)
- [ ] `scripts/train_classifier.py`
  - Mixed precision (`torch.cuda.amp`)
  - Gradient clipping (`max_norm=1.0`)
  - Early stopping on macro AUROC (patience=5)
  - Checkpoint: best val macro AUROC
  - Config from YAML only, no hardcoded values
- [ ] `scripts/evaluate.py` producing:
  - Per-class AUROC and AUPRC
  - Macro AUROC and Macro AUPRC
  - Sensitivity at 90% specificity per class
  - Output as `results/baseline_nih_eval.json`
- [ ] `tests/test_dataloader.py`: shape, dtype, label range checks
- [ ] `tests/test_model_shapes.py`: forward pass shape check

Exit condition (REQUIRED — blocks handoff):
- `make train` completes one full epoch without error
- `make eval` produces valid `results/baseline_nih_eval.json`
- All tests pass (`make test`)

Exit condition (TARGET — not a blocker):
- Macro AUROC > 0.70 on NIH val set
- If not reached, document in `DECISIONS.md` and proceed

Handoff: → Claude Opus 4.7 for Phase 2.

---

### PHASE 2 — Calibration + Abstention
**Owner: Claude Opus 4.7**
**Entry condition: Phase 1 exit conditions met. Checkpoint exists.**

Deliverables:
- [ ] `src/medguard/models/calibration.py`
  - Temperature scaling: one scalar T per class, fit on val set only
  - Isotonic regression as secondary calibrator
  - Platt scaling as tertiary option
  - Calibrator saved to disk (`calibrators/nih_temp_scaling.pkl`)
- [ ] `src/medguard/eval/calibration_metrics.py`
  - ECE with M=15 bins
  - MCE
  - Reliability diagram saved as `results/reliability_diagram.png`
  - Brier score per class
- [ ] `src/medguard/safety/abstention.py`
  - Per-class confidence threshold (NOT a single global threshold)
  - Rare/high-risk classes (Pneumothorax, Mass, Nodule) get lower threshold
  - Selective risk curve: accuracy vs coverage at varying thresholds
  - Returns `{prediction, confidence, abstained: bool, reason: str}`
- [ ] `scripts/calibrate.py`: fits calibrator on val, evaluates on test
- [ ] `tests/test_abstention.py`
- [ ] Architecture review of Phase 1 code (use Review block format):
  - Confirm sigmoid vs softmax
  - Confirm pos_weight calculation
  - Flag calibration-hostile patterns (e.g., BatchNorm after sigmoid)

**Critical implementation notes:**
- Temperature scaling fit on **val set only**. Never touch test set before final eval.
- Abstention rate must be reported alongside accuracy.
  Invariant: accuracy must improve (or stay flat) as abstention rate increases.
  If it does not, the threshold policy is wrong.
- Do NOT use a single global threshold across all 14 classes.

Exit condition (REQUIRED — blocks handoff):
- Calibration script runs without error
- Reliability diagram generated and committed
- ECE before and after calibration both reported in `results/calibration_report.json`
- No test-set data touched during calibration fitting
- `tests/test_abstention.py` passes

Exit condition (TARGET — not a blocker):
- Macro ECE < 0.10 post-calibration on NIH val set
- If not reached: ECE must at least improve over uncalibrated baseline; document gap in `DECISIONS.md`

Handoff: → Codex for Phase 3.

---

### PHASE 3 — Grounded Localization (VinDr-CXR)
**Owner: GPT-5.5 Codex**
**Entry condition: Phase 2 exit conditions met.**

Deliverables:
- [ ] `src/medguard/data/vindr.py`
  - 22 local abnormality labels + bounding box annotations
  - Consensus bbox from 17 radiologist annotations (majority vote or IoU merge)
  - Bounding boxes normalized to [0,1] relative to image size
- [ ] `src/medguard/explain/gradcam.py`
  - GradCAM targeting last conv layer before global average pool
  - GradCAM++ as secondary option
  - Returns heatmap as `np.ndarray` same size as input
  - Only runs when `confidence > abstention_threshold` (do not run on abstained predictions)
- [ ] `src/medguard/explain/overlays.py`
  - Overlay heatmap on original X-ray
  - Draw predicted bbox (red) and ground truth bbox (green)
  - Save 4×4 sample grid for README and notebooks
- [ ] `src/medguard/eval/localization_metrics.py`
  - IoU between CAM-derived bbox and ground truth bbox
  - mAP@0.5
  - Pointing game accuracy: is `argmax(CAM)` inside ground truth bbox?
- [ ] Sample overlay images in `notebooks/02_baseline_results.ipynb`

**Critical notes:**
- VinDr bboxes are in DICOM pixel space. Normalize before IoU.
- GradCAM target: last conv layer, NOT the FC head.
- Do NOT run GradCAM on every prediction. Gate on confidence threshold.

Exit condition (REQUIRED):
- `make eval` produces localization metrics JSON
- Sample overlays visually inspectable (not obviously broken)

Exit condition (TARGET — not a blocker):
- Pointing game accuracy > 0.50 on VinDr val set

Handoff: → Gemini for Phase 3B visual audit.

---

### PHASE 3B — Grad-CAM Visual Audit
**Owner: Gemini 3 Flash**
**Entry condition: Phase 3 overlay images committed.**

Deliverables:
- [ ] Visual inspection of ≥20 GradCAM overlay samples
- [ ] Report: is the model attending to correct anatomical regions?
- [ ] Failure mode catalog:
  - CAM firing on image border
  - CAM firing on L/R anatomical marker (very common failure — check explicitly)
  - CAM firing on pacemaker or tube artifact
  - CAM firing outside lung field entirely
- [ ] Per-class reliability: which findings have trustworthy localization?
- [ ] Go/no-go recommendation for Phase 4

Exit condition: Report in `notebooks/03_error_analysis.ipynb`.
Handoff: → Claude for Phase 4.

---

### PHASE 4 — VQA + Safety-Aware Assistant
**Design owner: Claude Opus 4.7 | Implementation owner: Codex**
**Entry condition: Phase 3B go/no-go is "go".**

Claude designs and specifies; Codex implements; Claude reviews.

Claude deliverables (design):
- [ ] Synthetic CXR-QA schema and generation script spec
- [ ] VQA answer format specification (question, answer, confidence, evidence, abstained)
- [ ] Abstention policy for VQA (when to refuse to answer)
- [ ] OOD gate specification: what gets rejected and why
- [ ] Safety banner copy for Gradio UI

Codex deliverables (implementation):
- [ ] `scripts/generate_vqa_dataset.py`: synthetic QA from NIH labels
- [ ] `src/medguard/models/vlm.py`: Qwen2.5-VL-3B-Instruct + QLoRA wiring
- [ ] `scripts/train_vlm_lora.py`: QLoRA fine-tune loop
- [ ] `src/medguard/safety/ood.py`: OOD detector (non-X-ray, lateral view, low-res)
- [ ] `api/app.py`: endpoints `POST /predict`, `POST /vqa`, `POST /explain`, `GET /health`
- [ ] `app/gradio_demo.py`: full demo with "NOT FOR CLINICAL USE" banner always visible

Claude review deliverables:
- [ ] Review of VQA answer quality on 20 examples
- [ ] Final `model_card.md` (complete)
- [ ] Final `datasheet.md`
- [ ] Final `README.md` with scorecard table and sample overlays

Exit condition (REQUIRED):
- `make demo` runs end-to-end
- OOD gate smoke tests (all must pass):
  - Natural image (e.g. cat photo) → rejected
  - Blank / all-black image → rejected or warned
  - Corrupted / noise-only image → rejected or warned
  - Lateral view X-ray (if view classifier available) → "unsupported view" warning
  - PA frontal X-ray → accepted (sanity check)
- VQA answers structured questions with confidence values
- "NOT FOR CLINICAL USE" banner visible in Gradio

Handoff: → Owner for final review and GitHub publication.

---

## 5. CONFLICT RESOLUTION PROTOCOL

Priority order when models disagree:

```
1. Safety constraint (Section 1) — always wins, no exceptions
2. Working core (pipeline runs)
3. Correct evaluation (metrics are valid)
4. Safety logic (abstention, OOD, thresholds)
5. Demo quality
6. Extensions (VLM, VQA)
```

If two models disagree on implementation of a shared file:
1. The **primary owner** (Section 2) decides.
2. Exception: if the disagreement involves a **hard constraint from Section 1**,
   Claude's position on safety wins regardless of file ownership.
3. Both positions and the resolution must be logged in `DECISIONS.md`.
4. Owner (Anıl) breaks ties that models cannot resolve.

---

## 6. REPO STRUCTURE (CANONICAL)

```
medguard-cxr/
├── configs/
│   ├── baseline_nih.yaml
│   ├── grounding_vindr.yaml
│   ├── vlm_lora.yaml
│   └── calibration.yaml
├── data/
│   ├── README.md          # download instructions only, no raw data
│   └── sample_manifest.csv
├── results/               # auto-generated, gitignored except .json summaries
├── calibrators/           # saved calibrator objects
├── src/medguard/
│   ├── __init__.py
│   ├── data/
│   │   ├── nih.py
│   │   ├── vindr.py
│   │   ├── dicom.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── classifier.py
│   │   ├── grounding.py
│   │   ├── vlm.py
│   │   └── calibration.py
│   ├── eval/
│   │   ├── classification_metrics.py
│   │   ├── localization_metrics.py
│   │   ├── calibration_metrics.py
│   │   └── report.py
│   ├── explain/
│   │   ├── gradcam.py
│   │   └── overlays.py
│   ├── api/
│   │   ├── app.py
│   │   └── schemas.py
│   └── safety/
│       ├── abstention.py
│       ├── ood.py
│       └── model_card.py
├── scripts/
│   ├── prepare_nih.py
│   ├── prepare_vindr.py
│   ├── generate_vqa_dataset.py
│   ├── train_classifier.py
│   ├── train_vlm_lora.py
│   ├── evaluate.py
│   ├── calibrate.py
│   └── launch_demo.py
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_baseline_results.ipynb
│   └── 03_error_analysis.ipynb
├── app/
│   └── gradio_demo.py
├── tests/
│   ├── test_dataloader.py
│   ├── test_model_shapes.py
│   ├── test_api_contract.py
│   └── test_abstention.py
├── DECISIONS.md
├── model_card.md
├── datasheet.md
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 7. TECH STACK (LOCKED — GET OWNER APPROVAL TO CHANGE)

| Layer | Tool | Version policy |
|-------|------|---------------|
| Medical preprocessing | MONAI, TorchIO, pydicom | pin in pyproject.toml |
| Baseline reference | TorchXRayVision | pin |
| Main CV backbone | DenseNet121 (default), EfficientNet/ViT (ablation) | pin torchvision |
| Localization | VinDr-CXR bbox + pytorch-grad-cam | pin |
| VLM | Qwen2.5-VL-3B-Instruct + QLoRA via PEFT | pin |
| Training | PyTorch Lightning or Accelerate | pin |
| Experiment tracking | v0.1: config YAML + fixed seeds; v0.2+: MLflow or W&B | — |
| Dataset versioning | Optional: DVC | — |
| Calibration | temperature-scaling library or custom | pin |
| Explainability | pytorch-grad-cam, Captum | pin |
| API | FastAPI + uvicorn | pin |
| Demo | Gradio | pin |
| Packaging | Docker, Makefile | — |
| CI | GitHub Actions | — |

---

## 8. EVALUATION SCORECARD

Fill this as phases complete. Commit updated scorecard after each phase.

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

---

## 9. DONE DEFINITION

Project is complete when ALL of the following are true:

- [ ] `make install && make prepare-nih && make train && make eval` runs clean
- [ ] Scorecard Section 8 fully filled with real numbers
- [ ] `make demo` launches Gradio with no errors
- [ ] OOD smoke tests pass: cat photo rejected, blank image rejected, PA X-ray accepted
- [ ] "NOT FOR CLINICAL USE" banner visible and non-removable in UI
- [ ] `model_card.md` covers: training data, eval data, intended use,
      out-of-scope use, ethical considerations, subgroup performance,
      calibration results, known failure modes
- [ ] `README.md` has: project pitch, scorecard table, 2–3 Grad-CAM
      overlay samples, reproduce instructions, disclaimer
- [ ] `DECISIONS.md` documents all non-trivial design choices
- [ ] All tests pass: `make test`

---

## 10. STARTING COMMAND FOR ANY NEW SESSION

Copy-paste this as the first message in any IDE / Codex / Claude / Gemini session:

```
Read MEDGUARD_MASTER_PROMPT.md fully before responding.

Then open your response with:
1. Handshake block (Section 3a)
2. Confirm which phase you are acting on
3. Confirm the phase entry condition is met
4. List the files you plan to touch this session

Do not write any code until steps 1–4 are complete.
If any entry condition is unclear or unmet, ask the owner before proceeding.
```

---

*v2.1 — Technical patch on GPT-5.5 Codex second review.
Five fixes applied:
(1) Sigmoid/BCEWithLogitsLoss clarified: model returns raw logits during training,
sigmoid applied at inference only — BCEWithLogitsLoss handles sigmoid internally.
(2) ImageNet normalization made conditional: ImageNet stats for ImageNet-pretrained
backbones; TorchXRayVision normalization if using XRV pretrained weights.
(3) DataLoader workers moved to config YAML: GPU default 4, CI/CPU 0, Mac 0–2.
(4) ECE < 0.10 demoted from blocker to target: blocker is now "ECE improves over
uncalibrated baseline + valid report generated".
(5) OOD smoke tests expanded: cat photo, blank image, corrupted image, lateral view,
PA sanity check — all must be formally tested.*
