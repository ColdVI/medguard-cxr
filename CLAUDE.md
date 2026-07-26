# CLAUDE.md — MedGuard-CXR

Context file for Claude Code. Read this before proposing or making any change.

---

## 1. What this project is

A research-oriented chest X-ray safety prototype. NIH ChestX-ray14 multi-label
classifier + post-hoc calibration + selective prediction (abstention) + rule-free
statistical OOD gating + Grad-CAM grounding evaluated cross-dataset on RSNA +
a safety-gated VQA layer.

**Current direction: this is an academic / theory-focused artifact, not a product.**
The deliverable that matters is `report/` (the LaTeX write-up) and the rigor of the
empirical claims — not deployment, not a demo, not API polish. Prioritize accordingly.

The owner is a CS undergraduate; the report is intended to read like a technical
research report with explicit research questions, related work, and honest negative
results.

---

## 2. Hard constraints — never violate

1. **Never overstate what ran.** The README has a "what actually ran" table. Every
   number in the report must trace to a real artifact in `results/`. If something was
   not executed, it is described as *not executed*, not as *implemented and working*.
2. **Never remove or weaken the claim-discipline material** — the what-actually-ran
   table, the what-not-to-claim list, `model_card.md`, `datasheet.md`. These are the
   spine of the project.
3. **No clinical claims.** No diagnostic utility, no clinical safety, no prospective
   validation, no "assists radiologists". Ever.
4. **Do not fabricate citations.** If a reference is needed and you are not certain of
   the exact authors/venue/year, flag it as `% TODO: verify` rather than inventing one.
   Fabricated BibTeX entries are the single worst failure mode for this project.
5. **Do not rewrite git history** and do not edit past entries in `DECISIONS.md`
   (it is append-only by design).
6. **Do not introduce role-based multi-agent workflow artifacts.** See §6.

---

## 3. Ground-truth results (do not contradict these)

| Component | Status | Result |
|---|---|---|
| NIH DenseNet121 classifier | ran | macro AUROC 0.8037, macro AUPRC 0.2685, n=25596 |
| Per-class temperature scaling | ran | ECE 0.3144 → 0.3113 (negligible improvement) |
| Isotonic / Platt calibration | implemented, not run in headline | — |
| RSNA cross-dataset grounding | ran | pointing-game 0.5138, mIoU low, mAP@0.5 = 0.0004 |
| Rule-based VQA | ran | exact match 0.6091, hallucination rate 0.0 |
| VLM (Qwen2.5-VL-3B) QLoRA fine-tune | code only, `enabled: false` | epochs_completed = 0 |
| VLM zero-shot eval | blocked (`bitsandbytes` / GPU) | never executed |

The near-zero mAP alongside a ~0.51 pointing-game score is **an expected finding, not a
bug** — it is the known strict-IoU vs. loose-hit-rate gap for CAM-style saliency, and it
should be framed as consistent with the literature (Saporta et al., *Nature Machine
Intelligence* 4:867–878, 2022).

The flat ECE is a **negative result and should be reported as one**, with the structural
explanation: temperature scaling is rank-preserving (a single positive scalar divisor
cannot change example ordering, only sharpness), and NIH labels are NLP-mined noisy
silver-standard labels, so the calibration target is itself noisy.

---

## 4. Method inventory (for accurate write-up)

- **Backbone**: DenseNet121, ImageNet-pretrained, 14-class multi-label head.
- **Loss**: `BCEWithLogitsLoss` on raw logits (numerically stable via log-sum-exp);
  `pos_weight = neg_count / pos_count` per class. Note in the report that pos_weight is
  reweighted MLE and therefore *itself* a source of miscalibration.
- **Calibration**: per-class temperature scaling (L-BFGS, strong Wolfe, minimizes
  validation NLL), per-class isotonic regression (PAV), per-class Platt scaling.
  Metrics: ECE/MCE with 15 equal-width bins, Brier score, reliability diagrams.
- **Explainability**: Grad-CAM (primary), Grad-CAM++ (secondary, higher-order
  derivatives, better for diffuse/multi-instance findings), last conv layer.
- **Selective prediction**: per-class two-sided threshold band (tau_lo, tau_hi);
  abstain inside the band. Risk-coverage curves reported both raw (non-monotone from
  finite-sample noise) and as a monotone envelope.
- **OOD**: learning-free, classifier-independent, three-stage — pixel std,
  2D FFT low-frequency energy ratio, RGB channel-difference + edge-orientation
  chi-square against a CXR edge prior. Deliberately avoids learned OOD scores so it does
  not inherit the classifier's own overconfidence.
- **VLM layer**: Qwen2.5-VL-3B-Instruct, 4-bit NF4, QLoRA (r=16, alpha=32,
  q/k/v/o_proj only). Positioned as a *presentation layer*, never a decision-maker:
  outputs must pass a classifier-consistency gate plus structured-JSON, banned-token,
  unsupported-finding, confidence, evidence, and length gates.
- **Data**: NIH ChestX-ray14 (classification), RSNA Pneumonia Detection 2018
  (grounding, real boxes, NIH Pneumonia → RSNA Lung Opacity mapping only),
  VinDr-CXR (deferred, inactive), synthetic template-generated VQA set.

### Design invariants

Properties the system holds by construction. They are describable in the report as
design decisions, and code changes must not silently drop them.

- **Uncertainty is always exposed.** Every prediction carries an explicit
  confidence/uncertainty value; no bare label is emitted without one.
- **Displayed positives carry visual evidence.** Any positive finding shown in the
  demo/UI is accompanied by a Grad-CAM overlay or box. A positive prediction is never
  presented on its own.
- **Runs are reproducible from config.** Fixed seed plus a YAML in `configs/` is the
  minimum; no experiment parameter should live only in a shell invocation.

---

## 5. Work backlog, in priority order

**P0 — `report/references.bib` has 3 entries (calibration-under-label-noise track:
`wu2026transts`, `frenkel2021classbased`, `li2022noisetransition`), 2 still marked
`% TODO: verify`. None of the three are `\cite`'d anywhere in `report/sections/` yet.
Still missing: the foundational core set — Wang et al. CVPR 2017 (ChestX-ray8);
Rajpurkar et al. 2017 (CheXNet); Guo et al. ICML 2017 (calibration); Selvaraju et al.
ICCV 2017 (Grad-CAM); Chattopadhay et al. WACV 2018 (Grad-CAM++); Saporta et al. NMI
2022 (saliency benchmark); Geifman & El-Yaniv NeurIPS 2017 (selective prediction);
Hu et al. 2021 (LoRA); Dettmers et al. 2023 (QLoRA); Mitchell et al. 2019 (model cards);
Gebru et al. 2018 (datasheets). Add these, verify each entry, and wire in the actual
`\cite` calls.**

**P0 — add explicit research questions to `report/sections/01_introduction.tex`.**
Suggested framing:
- RQ1: does per-class post-hoc calibration meaningfully reduce ECE under multi-label
  imbalance with noisy labels? (Answer: no — report as a negative result.)
- RQ2: how far do point-based and IoU-based grounding metrics diverge under
  cross-dataset transfer (NIH → RSNA)?
- RQ3: does a classifier-consistency gate suppress hallucination in a generative
  presentation layer?

**P0 — add a Related Work section.** The report currently has none.

**P1 — run isotonic and Platt calibration** (code already exists) and produce a
three-way comparison table against temperature scaling. Discuss the flexibility vs.
overfitting tradeoff given low positive counts in rare classes.

**P1 — bootstrap confidence intervals** for AUROC, AUPRC, and ECE. Point estimates
alone are weak for an academic write-up.

**P2 — formalize `03_methodology.tex`**: write the temperature-scaling objective,
ECE/MCE definitions, and pointing-game / IoU / mAP definitions as equations.

**P2 — optional**: run VLM zero-shot on GPU (Colab) against the rule-based baseline.
Only if the environment is available; otherwise leave clearly marked as not executed.

---

## 6. Working protocol

**One agent, one session, one owner.** You are a coding assistant working with the
repository owner directly. There is no implementer/reviewer/auditor split, no phase
gates, no GO/NO_GO verdicts, no agent role assignment. That structure belonged to an
earlier stage of the project and has been retired.

Concretely:

- Do **not** create new governance/protocol/prompt markdown files (no successor to
  `MEDGUARD_MASTER_PROMPT.md`, no role charters, no phase plans).
- Do **not** append role-attributed entries to `DECISIONS.md`. If a decision needs
  recording, it goes in the owner's own words, or in a plain git commit message.
- Do **not** write "reviewed by", "audited by", "verdict:", "Phase X gate", or
  similar language into any file.
- Do **not** add `Co-Authored-By:` trailers or "Generated with Claude Code" lines to
  commit messages. Commit messages are the owner's, in plain imperative style.
- Work in **small, reviewable increments**: one concern per change, stop and show the
  diff, wait for the owner to read it before moving on. The owner must be able to
  explain every line to a third party, so do not batch large multi-file rewrites.
- When a design choice has a real tradeoff, say what the tradeoff is rather than just
  picking one. The owner needs the reasoning, not only the result.

---

## 7. Repo conventions

- Python, `ruff` for lint, `pytest` for tests, pinned deps in `pyproject.toml`.
- Pipeline entry points via `Makefile` (`make train`, `eval`, `calibrate`,
  `eval-grounding-rsna`).
- CI runs import smoke tests + ruff only; the full suite requires data/GPU and is run
  locally. Do not add CI steps that will fail without data.
- Configs live in `configs/`, results as JSON in `results/`.
- Tests assert real statistical properties (e.g. temperature converges to T→1 on
  perfectly calibrated synthetic logits). Keep that standard — no trivial assertions.
