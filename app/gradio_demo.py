"""Local Gradio demo for Phase 4 safety-aware MedGuard-CXR."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from medguard.api.app import MedGuardService
from medguard.api.schemas import SAFETY_DISCLAIMER, SMOKE_WARNING

PHASE = "4"

BANNER_HTML = """
<div style="background:#7f1d1d; color:#fff; padding:12px; font-weight:700;
            text-align:center; border-bottom:3px solid #fca5a5;">
  ⚠️ NOT FOR CLINICAL USE — research demo only ⚠️<br/>
  <span style="font-weight:400; font-size:0.9em;">
    MedGuard-CXR does not provide diagnoses, treatment, or medical advice.
  </span>
</div>
"""
SMOKE_SUBSCRIPT_HTML = (
    "<div style='color:#7f1d1d; font-weight:600; padding:6px 0;'>"
    "Running in SMOKE mode — outputs are not derived from validated weights."
    "</div>"
)


def is_available() -> bool:
    """Return whether the Phase 4 Gradio demo is implemented."""

    return True


def build_demo(service: MedGuardService | None = None) -> Any:
    """Build a local-only Gradio Blocks demo."""

    import gradio as gr

    resolved_service = service or MedGuardService()
    with gr.Blocks(title="MedGuard-CXR") as demo:
        gr.HTML(BANNER_HTML)
        if resolved_service.provenance.is_smoke:
            gr.HTML(SMOKE_SUBSCRIPT_HTML)
        with gr.Row():
            image = gr.Image(type="pil", label="Image Upload")
            predictions = gr.Dataframe(
                headers=["Class", "Conf", "Decision"],
                label="Predictions",
                interactive=False,
            )
        evidence = gr.Image(label="Evidence", visible=False)
        question = gr.Textbox(label="Ask a question", value="Is there evidence of Pneumothorax?")
        answer = gr.Textbox(label="Answer", interactive=False)
        gr.Markdown(f"**{SAFETY_DISCLAIMER}**")

        def run_predict(
            input_image: Image.Image | np.ndarray | None,
        ) -> tuple[list[list[str]], Any]:
            if input_image is None:
                return [], gr.update(visible=False)
            response = resolved_service.predict(_ensure_pil(input_image))
            rows = []
            evidence_visible = False
            for prediction in response.predictions or []:
                if prediction.abstained:
                    decision = f"ABSTAIN - {prediction.reason}"
                    confidence = ""
                else:
                    decision = "positive" if prediction.prediction == 1 else "negative"
                    confidence = f"{prediction.confidence:.2f}"
                    evidence_visible = evidence_visible or prediction.prediction == 1
                rows.append([prediction.class_name, confidence, decision])
            return rows, gr.update(visible=evidence_visible)

        def run_vqa(input_image: Image.Image | np.ndarray | None, text: str) -> str:
            if input_image is None:
                return SAFETY_DISCLAIMER
            response = resolved_service.vqa(_ensure_pil(input_image), text)
            reason = response.reason or "-"
            abstained = "Yes" if response.abstained else "No"
            return (
                f'{response.answer}\n\nConfidence: {response.confidence:.2f} | '
                f"Abstained: {abstained} | Reason: {reason}"
            )

        image.change(run_predict, inputs=image, outputs=[predictions, evidence])
        question.submit(run_vqa, inputs=[image, question], outputs=answer)
    return demo


def banner_html() -> str:
    """Expose the non-removable banner constant for tests."""

    return BANNER_HTML


def evidence_panel_visible(predictions: list[dict[str, Any]]) -> bool:
    """Return whether any non-abstained positive prediction has eligible evidence."""

    return any(
        item.get("prediction") == 1 and not item.get("abstained") for item in predictions
    )


def main() -> None:
    """Launch the local demo."""

    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860)


def _ensure_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")


__all__ = [
    "BANNER_HTML",
    "SMOKE_SUBSCRIPT_HTML",
    "SMOKE_WARNING",
    "banner_html",
    "build_demo",
    "evidence_panel_visible",
    "is_available",
    "main",
]


if __name__ == "__main__":
    main()
