"""Phase 4 Gradio demo safety tests."""

import inspect

import numpy as np
from app import gradio_demo

from medguard.api.app import MedGuardService
from medguard.data.nih import NIH_LABELS


def test_banner_html_appears_in_rendered_blocks() -> None:
    demo = gradio_demo.build_demo()

    assert "NOT FOR CLINICAL USE" in gradio_demo.banner_html()
    assert demo is not None


def test_banner_text_is_a_python_constant() -> None:
    source = inspect.getsource(gradio_demo)

    assert "BANNER_HTML = " in source
    assert "NOT FOR CLINICAL USE" in gradio_demo.BANNER_HTML


def test_evidence_panel_hidden_when_all_abstained() -> None:
    assert gradio_demo.evidence_panel_visible([{"prediction": None, "abstained": True}]) is False


def test_evidence_panel_shown_for_positive_with_eligible_explanation() -> None:
    assert gradio_demo.evidence_panel_visible([{"prediction": 1, "abstained": False}]) is True


def test_smoke_mode_subscript_visible_when_provenance_is_smoke() -> None:
    assert "SMOKE mode" in gradio_demo.SMOKE_SUBSCRIPT_HTML


def test_demo_starts_without_real_checkpoint() -> None:
    probs = np.full(len(NIH_LABELS), 0.5, dtype=np.float64)
    demo = gradio_demo.build_demo(service=MedGuardService(fixed_probabilities=probs))

    assert demo is not None
