"""Typed schema for cross-dataset chest X-ray label mappings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MappingType(str, Enum):
    """Permitted semantic relationships between source and canonical labels."""

    EXACT = "exact"
    APPROXIMATE = "approximate"
    BROADER = "broader"
    NARROWER = "narrower"
    UNSUPPORTED = "unsupported"


class MappingConfidence(str, Enum):
    """Human-reviewed confidence assigned to one mapping."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class LabelMapping:
    """One explicit dataset-label to canonical-label relationship."""

    canonical_label: str
    source_dataset: str
    source_label: str
    mapping_type: MappingType
    mapping_confidence: MappingConfidence
    include_in_cross_dataset_metrics: bool
    notes: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LabelMapping:
        """Parse and validate a mapping from YAML-compatible data."""

        required = {
            "canonical_label",
            "source_dataset",
            "source_label",
            "mapping_type",
            "mapping_confidence",
            "include_in_cross_dataset_metrics",
            "notes",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"Ontology mapping is missing fields: {', '.join(missing)}")
        mapping = cls(
            canonical_label=str(value["canonical_label"]).strip(),
            source_dataset=str(value["source_dataset"]).strip().lower(),
            source_label=str(value["source_label"]).strip(),
            mapping_type=MappingType(value["mapping_type"]),
            mapping_confidence=MappingConfidence(value["mapping_confidence"]),
            include_in_cross_dataset_metrics=bool(
                value["include_in_cross_dataset_metrics"]
            ),
            notes=str(value["notes"]).strip(),
        )
        if not mapping.canonical_label or not mapping.source_dataset or not mapping.source_label:
            raise ValueError("Ontology labels and source_dataset must be non-empty")
        if (
            mapping.mapping_type is not MappingType.EXACT
            and mapping.include_in_cross_dataset_metrics
        ):
            raise ValueError(
                "Only exact mappings may be enabled for primary cross-dataset metrics"
            )
        return mapping
