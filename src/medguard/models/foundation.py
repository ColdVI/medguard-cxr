"""External chest X-ray foundation model access contracts."""

from __future__ import annotations


class FoundationModelAccessError(RuntimeError):
    """External terms, credentials, or artifacts prevent model construction."""


def build_eva_x_s(*, checkpoint: str | None, mode: str) -> None:
    """Validate EVA-X request without silently substituting another backbone."""

    if mode not in {"linear_probe", "full_finetune"}:
        raise ValueError(f"Unsupported EVA-X mode: {mode}")
    if not checkpoint:
        raise FoundationModelAccessError(
            "EVA-X-S checkpoint is unavailable; status=blocked_external_access"
        )
    raise NotImplementedError("EVA-X checkpoint integration is scheduled after access verification")


def build_cxr_foundation_probe(*, model_path: str | None, probe: str) -> None:
    """Validate CXR Foundation probe selection without fabricating embeddings."""

    if probe not in {"logistic_regression", "mlp_2layer"}:
        raise ValueError(f"Unsupported CXR Foundation probe: {probe}")
    if not model_path:
        raise FoundationModelAccessError(
            "CXR Foundation model is unavailable; status=blocked_external_access"
        )
    raise NotImplementedError("CXR Foundation backend requires the licensed model package")
