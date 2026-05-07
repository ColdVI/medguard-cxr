# MEDGUARD-CXR Decisions Log

This log is append-only. Record non-trivial design choices, safety decisions, and owner conflict resolutions here.

| Date | Phase | Decision | Rationale | Owner |
|------|-------|----------|-----------|-------|
| 2026-05-07 | 0A review fixes | Added explicit `__init__.py` files to all `src/medguard` subpackages. | Non-editable wheels must include every subpackage; explicit packages avoid relying on PEP 420 namespace behavior. | Codex |
| 2026-05-07 | 0A review fixes | Collapsed Python support to `>=3.10,<3.13` and removed the untested Python 3.13 dependency branch. | CI and Docker validate Python 3.11 only; shipping two unverified dependency matrices would create packaging risk. | Codex |
| 2026-05-07 | 0A review fixes | Kept the Docker GPU target as an intentional Phase 0A placeholder deferred until Phase 1. | GPU install and CUDA runtime validation belong with the first real training path, not the import-safe skeleton. | Codex |
| 2026-05-07 | 0A review fixes | Deferred `bitsandbytes`, `timm`, and v0.2 experiment tracking packages until their owning phases. | QLoRA, ablations, and MLflow/W&B are not used in Phase 0A; owners must pin them before first use. | Codex |
| 2026-05-07 | 0A review fixes | Kept `tests/test_imports.py` as a deliberate canonical-structure extension for CI smoke coverage. | Section 6 omits an import-smoke test, but the Phase 0A task explicitly requested one. | Codex |
| 2026-05-07 | 0A review fixes | Assigned `src/medguard/models/grounding.py` ownership to Codex with Claude review. | Section 6 includes the file but Section 2 omits ownership; this aligns it with Phase 3 localization work. | Codex |
| 2026-05-07 | 0A review fixes | Assigned `notebooks/02_baseline_results.ipynb` ownership to Codex with Claude review. | The notebook is populated by Phase 1 baseline results and needs an explicit owner. | Codex |
| 2026-05-07 | 0A review fixes | Documented the Phase 0A placeholder convention: `PHASE` plus `is_available() -> False`. | The temporary skeleton API is explicit until each owning phase replaces or removes it. | Codex |
