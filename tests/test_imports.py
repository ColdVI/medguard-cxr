"""Import smoke tests for the Phase 0A package skeleton."""

import importlib

MODULES = [
    "medguard",
    "medguard.data.nih",
    "medguard.data.vindr",
    "medguard.data.dicom",
    "medguard.data.transforms",
    "medguard.models.classifier",
    "medguard.models.grounding",
    "medguard.models.vlm",
    "medguard.models.calibration",
    "medguard.eval.classification_metrics",
    "medguard.eval.localization_metrics",
    "medguard.eval.calibration_metrics",
    "medguard.eval.report",
    "medguard.explain.gradcam",
    "medguard.explain.overlays",
    "medguard.api.app",
    "medguard.api.schemas",
    "medguard.safety.abstention",
    "medguard.safety.ood",
    "medguard.safety.model_card",
]


def test_import_medguard() -> None:
    """The top-level package imports cleanly."""
    module = importlib.import_module("medguard")
    assert module.__version__ == "0.0.0"


def test_import_skeleton_modules() -> None:
    """All skeleton modules import without optional runtime dependencies."""
    for module_name in MODULES:
        importlib.import_module(module_name)
