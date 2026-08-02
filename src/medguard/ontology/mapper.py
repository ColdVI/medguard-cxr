"""Loading and safe-intersection logic for the Academic V2 label ontology."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from medguard.ontology.schema import LabelMapping, MappingConfidence, MappingType


@dataclass(frozen=True)
class CommonLabelSelection:
    """Safe common labels and auditable exclusion reasons."""

    included: tuple[str, ...]
    excluded: dict[str, str]


class LabelOntology:
    """Validated collection of explicit source-label mappings."""

    def __init__(
        self,
        mappings: list[LabelMapping],
        canonical_labels: list[str],
        version: str,
    ) -> None:
        self.mappings = tuple(mappings)
        self.canonical_labels = tuple(canonical_labels)
        self.version = version
        self._by_source: dict[tuple[str, str], list[LabelMapping]] = {}
        for mapping in self.mappings:
            key = (mapping.source_dataset, mapping.source_label)
            if mapping.canonical_label not in self.canonical_labels:
                raise ValueError(
                    f"Unknown canonical label {mapping.canonical_label!r} in mapping {key}"
                )
            existing = self._by_source.setdefault(key, [])
            if any(item.canonical_label == mapping.canonical_label for item in existing):
                raise ValueError(
                    f"Duplicate ontology source-to-canonical mapping: {key} -> "
                    f"{mapping.canonical_label}"
                )
            existing.append(mapping)

    def map_label(self, dataset: str, source_label: str) -> LabelMapping | None:
        """Return the explicit mapping; never infer from similar label spelling."""

        candidates = self._by_source.get((dataset.lower(), source_label), [])
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda mapping: (
                mapping.mapping_type is not MappingType.EXACT,
                not mapping.include_in_cross_dataset_metrics,
            ),
        )[0]

    def safe_common_labels(self, datasets: list[str]) -> CommonLabelSelection:
        """Select labels with exact, high-confidence, enabled mappings in every dataset."""

        normalized = tuple(dict.fromkeys(dataset.lower() for dataset in datasets))
        included: list[str] = []
        excluded: dict[str, str] = {}
        for canonical in self.canonical_labels:
            per_dataset = {
                dataset: [
                    mapping
                    for mapping in self.mappings
                    if mapping.source_dataset == dataset
                    and mapping.canonical_label == canonical
                ]
                for dataset in normalized
            }
            missing = [dataset for dataset, values in per_dataset.items() if not values]
            if missing:
                excluded[canonical] = "missing mapping for: " + ", ".join(missing)
                continue
            unsafe = [
                dataset
                for dataset, values in per_dataset.items()
                if not any(_is_primary_safe(mapping) for mapping in values)
            ]
            if unsafe:
                excluded[canonical] = "no exact high-confidence primary mapping for: " + ", ".join(
                    unsafe
                )
                continue
            included.append(canonical)
        return CommonLabelSelection(tuple(included), excluded)


def load_ontology(path: str | Path) -> LabelOntology:
    """Load a YAML ontology and validate every entry."""

    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("mappings"), list):
        raise ValueError("Ontology YAML must contain a mappings list")
    canonical = raw.get("canonical_labels")
    if not isinstance(canonical, list) or not canonical:
        raise ValueError("Ontology YAML must contain canonical_labels")
    mappings = [LabelMapping.from_dict(item) for item in raw["mappings"]]
    return LabelOntology(mappings, [str(item) for item in canonical], str(raw.get("version", "")))


def _is_primary_safe(mapping: LabelMapping) -> bool:
    return (
        mapping.mapping_type is MappingType.EXACT
        and mapping.mapping_confidence is MappingConfidence.HIGH
        and mapping.include_in_cross_dataset_metrics
    )
