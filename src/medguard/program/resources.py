"""Deterministic Colab resource profile selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    precision: str
    max_resolution: int
    vlm_model: str
    detector_batch_size: int


def select_resource_profile(vram_gb: float) -> ResourceProfile:
    if vram_gb >= 40:
        return ResourceProfile("a100", "bf16", 384, "qwen3_vl_4b", 4)
    if vram_gb >= 22:
        return ResourceProfile("l4", "fp16", 384, "qwen3_vl_4b", 2)
    return ResourceProfile("t4", "fp16", 224, "qwen3_vl_2b", 1)
