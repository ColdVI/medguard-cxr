"""Import smoke tests for the Phase 0A package skeleton."""

import importlib

PACKAGE_MODULES = [
    "medguard",
    "medguard.data",
    "medguard.models",
    "medguard.eval",
    "medguard.explain",
    "medguard.api",
    "medguard.safety",
]

IMPLEMENTED_MODULES = [
    "medguard.data.nih",
    "medguard.data.transforms",
    "medguard.models.classifier",
]

SKELETON_MODULES = [
    "medguard.data.vindr",
    "medguard.data.dicom",
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

MODULES = PACKAGE_MODULES + IMPLEMENTED_MODULES + SKELETON_MODULES


def test_import_medguard() -> None:
    """The top-level package imports cleanly."""
    module = importlib.import_module("medguard")
    assert module.__version__ == "0.0.0"


def test_import_skeleton_modules() -> None:
    """All skeleton modules import without optional runtime dependencies."""
    for module_name in MODULES:
        importlib.import_module(module_name)


def test_skeleton_modules_are_not_available() -> None:
    """Phase 0A placeholder modules advertise that real logic is not available."""
    for module_name in SKELETON_MODULES:
        module = importlib.import_module(module_name)
        assert module.is_available() is False


def test_phase1_modules_are_available() -> None:
    """Phase 1 modules advertise that real baseline logic is available."""
    for module_name in IMPLEMENTED_MODULES:
        module = importlib.import_module(module_name)
        assert module.is_available() is True
