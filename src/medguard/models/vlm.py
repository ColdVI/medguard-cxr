"""Optional Phase 4B VLM inference and safety filtering.

The calibrated classifier remains the decision-maker. This module only accepts a
VLM answer when it is structured, non-clinical, classifier-consistent, and safe
to display. Otherwise callers must fall back to rule-based VQA.
"""

from __future__ import annotations

import importlib.util
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from medguard.api.schemas import (
    SAFETY_DISCLAIMER,
    EvidencePayload,
    ModelProvenance,
    VQAResponse,
    VQASource,
)
from medguard.data.nih import NIH_LABELS
from medguard.safety.abstention import AbstentionThresholds, PredictionRecord
from medguard.vqa.rule_based import answer_question
from medguard.vqa.templates import BANNED_ANSWER_TOKENS, display_finding

PHASE = "4B"
VLM_SOURCES: tuple[VQASource, VQASource] = ("vlm_zero_shot", "vlm_lora")
REQUIRED_OUTPUT_FIELDS = {
    "question",
    "answer",
    "confidence",
    "evidence",
    "abstained",
    "reason",
    "safety_disclaimer",
    "source",
}
UNSUPPORTED_FINDING_TOKENS = [
    "brain tumor",
    "tumor",
    "cancer",
    "fracture",
    "kidney stone",
    "kidney stones",
    "covid",
    "ct scan",
    "mri",
    "ultrasound",
]
MAX_ANSWER_TOKENS = 200


class VLMUnavailableError(RuntimeError):
    """Raised when optional VLM dependencies or runtime resources are absent."""


@dataclass(frozen=True)
class VLMFilterResult:
    """Post-inference safety filter outcome."""

    passed: bool
    filter_result: Literal["pass", "reject"]
    reason: str
    raw_vlm_answer: str
    filtered_answer: str | None
    parsed_payload: dict[str, Any] | None = None


class VLMInferenceEngine:
    """Thin wrapper around an optional Qwen2.5-VL base model and LoRA adapter."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        source_tag: VQASource = "vlm_zero_shot",
        adapter_path: str | Path | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        self.model = model
        self.processor = processor
        self.source_tag: VQASource = source_tag
        self.adapter_path = Path(adapter_path) if adapter_path is not None else None
        self.max_new_tokens = max_new_tokens
        self.adapter_loaded = source_tag == "vlm_lora"

    def generate(self, image: Image.Image, question: str, system_prompt: str) -> str:
        """Greedy-decode a structured JSON answer for one image/question pair."""

        prompt = _prompt(system_prompt, question)
        inputs = self.processor(
            text=prompt,
            images=image.convert("RGB"),
            return_tensors="pt",
        )
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return str(decoded[0] if decoded else "")


def is_available() -> bool:
    """Return whether optional VLM dependencies are importable."""

    return not dependency_status()["missing"]


def dependency_status() -> dict[str, list[str]]:
    """Report optional Phase 4B dependency availability without importing them."""

    required = ["torch", "transformers", "peft", "bitsandbytes"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    return {
        "required": required,
        "missing": missing,
        "available": [name for name in required if name not in missing],
    }


def load_vlm(config: Mapping[str, Any]) -> VLMInferenceEngine:
    """Load the base VLM and optional LoRA adapter from config.

    This performs heavy imports only after availability has been checked.
    """

    if not is_available():
        missing = ", ".join(dependency_status()["missing"])
        raise VLMUnavailableError(f"Optional VLM dependencies unavailable: {missing}")

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:  # pragma: no cover - depends on installed transformers.
        from transformers import AutoModelForVision2Seq as ModelClass

    vlm_cfg = dict(config.get("vlm", {}))
    model_name = str(vlm_cfg.get("base_model", "Qwen/Qwen2.5-VL-3B-Instruct"))
    zero_shot_cfg = dict(vlm_cfg.get("zero_shot", {}))
    quant_cfg = dict(vlm_cfg.get("quantization", {}))
    quantization_config = None
    if bool(vlm_cfg.get("load_in_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(
                quant_cfg.get("bnb_4bit_use_double_quant", True)
            ),
            bnb_4bit_compute_dtype=torch.float16,
        )

    processor = AutoProcessor.from_pretrained(model_name)
    model = ModelClass.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )

    adapter_cfg = dict(vlm_cfg.get("adapter", {}))
    adapter_path = Path(str(adapter_cfg.get("path", "checkpoints/vlm_lora_adapter")))
    source_tag: VQASource = "vlm_zero_shot"
    if adapter_path.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        source_tag = "vlm_lora"

    return VLMInferenceEngine(
        model=model,
        processor=processor,
        source_tag=source_tag,
        adapter_path=adapter_path if source_tag == "vlm_lora" else None,
        max_new_tokens=int(zero_shot_cfg.get("max_new_tokens", 128)),
    )


def phase4b_system_prompt(class_names: list[str] | None = None) -> str:
    """Return Claude's constrained Phase 4B system prompt."""

    findings = ", ".join(class_names or NIH_LABELS)
    return (
        "You are a research chest X-ray analysis assistant. You are NOT a clinical "
        "diagnostic system. You may ONLY answer questions about these 14 findings: "
        f"{findings}. If asked for a diagnosis, treatment, or clinical advice, refuse. "
        "If uncertain, say you are not confident enough to answer. Return only JSON "
        f"and set safety_disclaimer exactly to: {json.dumps(SAFETY_DISCLAIMER)}"
    )


def parse_vlm_output(raw_output: str | Mapping[str, Any]) -> dict[str, Any]:
    """Parse a VLM JSON/dict answer, including fenced JSON blocks."""

    if isinstance(raw_output, Mapping):
        return dict(raw_output)
    text = str(raw_output).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced is not None:
        text = fenced.group(1)
    elif "{" in text and "}" in text:
        text = text[text.index("{") : text.rindex("}") + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("VLM output must be structured JSON or a parseable dict.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("VLM output JSON must decode to an object.")
    return parsed


def validate_vlm_payload(
    payload: Mapping[str, Any],
    expected_question: str | None = None,
    expected_source: VQASource | None = None,
) -> dict[str, Any]:
    """Validate the required Phase 4B structured VQA fields."""

    parsed = dict(payload)
    missing = REQUIRED_OUTPUT_FIELDS - set(parsed)
    if missing:
        raise ValueError(f"VLM payload missing required fields: {sorted(missing)}")
    if expected_question is not None and parsed["question"] != expected_question:
        raise ValueError("VLM payload question does not match the requested question.")
    if parsed["safety_disclaimer"] != SAFETY_DISCLAIMER:
        raise ValueError("VLM payload must carry the canonical safety disclaimer.")
    if parsed["source"] not in VLM_SOURCES:
        raise ValueError("VLM payload source must be vlm_zero_shot or vlm_lora.")
    if expected_source is not None and parsed["source"] != expected_source:
        raise ValueError("VLM payload source does not match the active VLM backend.")
    confidence = float(parsed["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("VLM payload confidence must be in [0, 1].")
    return parsed


def filter_vlm_payload(
    payload: Mapping[str, Any],
    classifier_decision: PredictionRecord,
    evidence: EvidencePayload | None,
    max_answer_tokens: int = MAX_ANSWER_TOKENS,
) -> VLMFilterResult:
    """Run Claude's post-inference VLM safety filters."""

    try:
        parsed = validate_vlm_payload(payload)
    except ValueError as exc:
        return _reject(str(exc), str(payload), None)

    answer = str(parsed["answer"])
    if len(answer.split()) > max_answer_tokens:
        return _reject("length_gate", answer, parsed)
    banned = _first_banned_token(answer)
    if banned is not None:
        return _reject(f"banned_token:{banned}", answer, parsed)
    if mentions_unsupported_finding(answer):
        return _reject("hallucination_unsupported_finding", answer, parsed)
    if classifier_decision.abstained:
        return _reject("confidence_gate", answer, parsed)
    if not check_consistency(answer, classifier_decision):
        return _reject("classifier_conflict", answer, parsed)
    if _answer_direction(answer, classifier_decision.class_name) == "positive" and evidence is None:
        return _reject("evidence_unavailable", answer, parsed)
    return VLMFilterResult(
        passed=True,
        filter_result="pass",
        reason="",
        raw_vlm_answer=answer,
        filtered_answer=answer,
        parsed_payload=parsed,
    )


def vlm_response_or_fallback(
    raw_output: str | Mapping[str, Any],
    question: str,
    classifier_decision: PredictionRecord,
    provenance: ModelProvenance,
    evidence: EvidencePayload | None,
    fallback: VQAResponse,
    source: VQASource,
    max_answer_tokens: int = MAX_ANSWER_TOKENS,
) -> tuple[VQAResponse, VLMFilterResult]:
    """Return a safe VLM response or a rule-based fallback response."""

    try:
        payload = parse_vlm_output(raw_output)
        payload = validate_vlm_payload(
            payload,
            expected_question=question,
            expected_source=source,
        )
    except ValueError as exc:
        return _fallback_response(fallback, str(exc)), _reject(str(exc), str(raw_output), None)

    result = filter_vlm_payload(
        payload,
        classifier_decision=classifier_decision,
        evidence=evidence,
        max_answer_tokens=max_answer_tokens,
    )
    if not result.passed:
        return _fallback_response(fallback, result.reason), result

    confidence = float(classifier_decision.confidence)
    return (
        VQAResponse(
            question=question,
            answer=str(payload["answer"]),
            confidence=confidence,
            evidence=evidence if classifier_decision.prediction == 1 else None,
            abstained=False,
            reason="",
            safety_disclaimer=SAFETY_DISCLAIMER,
            model_provenance=provenance,
            source=source,
        ),
        result,
    )


def answer_with_optional_vlm(
    image: Image.Image,
    question: str,
    probabilities: np.ndarray,
    thresholds: AbstentionThresholds,
    provenance: ModelProvenance,
    evidence: EvidencePayload | None,
    vlm_engine: VLMInferenceEngine | None,
) -> tuple[VQAResponse, VLMFilterResult | None]:
    """Use VLM as a presentation layer when it is loaded and safe."""

    fallback = answer_question(
        question=question,
        probabilities=probabilities,
        thresholds=thresholds,
        provenance=provenance,
        evidence=evidence,
        require_evidence_for_positive=True,
    )
    if vlm_engine is None or fallback.abstained:
        return fallback, None
    class_name = (
        fallback.evidence.class_name
        if fallback.evidence is not None
        else _class_from_question(question)
    )
    if class_name is None:
        return fallback, None
    decision = _decision_from_response(fallback, class_name)
    started = time.perf_counter()
    raw = vlm_engine.generate(image, question, phase4b_system_prompt(list(thresholds.classes)))
    response, result = vlm_response_or_fallback(
        raw_output=raw,
        question=question,
        classifier_decision=decision,
        provenance=provenance,
        evidence=evidence,
        fallback=fallback,
        source=vlm_engine.source_tag,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    if result.parsed_payload is not None:
        result.parsed_payload["latency_ms"] = latency_ms
    return response, result


def check_consistency(vlm_answer: str, classifier_decision: PredictionRecord) -> bool:
    """Return whether a VLM answer agrees with the calibrated classifier decision."""

    direction = _answer_direction(vlm_answer, classifier_decision.class_name)
    if classifier_decision.abstained:
        return direction in {"negative", "uncertain"}
    if classifier_decision.prediction == 1:
        return direction == "positive"
    if classifier_decision.prediction == 0:
        return direction in {"negative", "uncertain"}
    return False


def mentions_unsupported_finding(answer: str) -> bool:
    """Return whether answer text mentions unsupported medical findings."""

    lowered = answer.lower()
    allowed = {display_finding(label).lower() for label in NIH_LABELS}
    for token in UNSUPPORTED_FINDING_TOKENS:
        if token in lowered and token not in allowed:
            return True
    return False


def _answer_direction(answer: str, class_name: str) -> Literal["positive", "negative", "uncertain"]:
    lowered = answer.lower()
    finding = display_finding(class_name).lower()
    if "not confident" in lowered or "not confident enough" in lowered:
        return "uncertain"
    if "does not detect" in lowered or "no visible evidence" in lowered:
        return "negative"
    if f"consistent with {finding}" in lowered or f"evidence of {finding}" in lowered:
        return "positive"
    return "uncertain"


def _first_banned_token(answer: str) -> str | None:
    lowered = answer.lower()
    for token in BANNED_ANSWER_TOKENS:
        if token.lower() in lowered:
            return token
    return None


def _reject(
    reason: str,
    raw_answer: str,
    payload: dict[str, Any] | None,
) -> VLMFilterResult:
    return VLMFilterResult(
        passed=False,
        filter_result="reject",
        reason=reason,
        raw_vlm_answer=raw_answer,
        filtered_answer=None,
        parsed_payload=payload,
    )


def _fallback_response(fallback: VQAResponse, rejection_reason: str = "") -> VQAResponse:
    reason = fallback.reason
    if not fallback.abstained and rejection_reason:
        reason = "vlm_output_rejected"
    return VQAResponse(
        question=fallback.question,
        answer=fallback.answer,
        confidence=fallback.confidence,
        evidence=fallback.evidence,
        abstained=fallback.abstained,
        reason=reason,
        safety_disclaimer=fallback.safety_disclaimer,
        model_provenance=fallback.model_provenance,
        source="rule_based",
    )


def _decision_from_response(response: VQAResponse, class_name: str) -> PredictionRecord:
    if response.abstained:
        prediction: int | None = None
    elif response.evidence is not None:
        prediction = 1
    elif "does not detect" in response.answer.lower():
        prediction = 0
    else:
        prediction = None
    return PredictionRecord(
        image_id=None,
        class_name=class_name,
        prediction=prediction,
        confidence=response.confidence,
        abstained=response.abstained,
        reason=response.reason,
    )


def _class_from_question(question: str) -> str | None:
    lowered = question.replace("_", " ").lower()
    for class_name in NIH_LABELS:
        if display_finding(class_name).lower() in lowered:
            return class_name
    return None


def _prompt(system_prompt: str, question: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        "<image>\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


__all__ = [
    "MAX_ANSWER_TOKENS",
    "REQUIRED_OUTPUT_FIELDS",
    "VLMFilterResult",
    "VLMInferenceEngine",
    "VLMUnavailableError",
    "answer_with_optional_vlm",
    "check_consistency",
    "dependency_status",
    "filter_vlm_payload",
    "is_available",
    "load_vlm",
    "mentions_unsupported_finding",
    "parse_vlm_output",
    "phase4b_system_prompt",
    "validate_vlm_payload",
    "vlm_response_or_fallback",
]
