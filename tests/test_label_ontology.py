"""Academic V2 label ontology safety tests."""

import pytest

from medguard.ontology.mapper import load_ontology
from medguard.ontology.schema import LabelMapping

ONTOLOGY = "configs/ontology/cxr_labels.yaml"


def test_ontology_never_silently_equates_prohibited_pairs() -> None:
    ontology = load_ontology(ONTOLOGY)

    assert ontology.map_label("nih", "Infiltration").mapping_type.value == "approximate"
    assert not ontology.map_label(
        "nih", "Infiltration"
    ).include_in_cross_dataset_metrics
    assert ontology.map_label("nih", "Mass").canonical_label == "Lung Lesion"
    assert ontology.map_label("rsna", "Lung Opacity").canonical_label == "Lung Opacity"
    pneumonia = [
        item
        for item in ontology.mappings
        if item.source_dataset == "rsna" and item.canonical_label == "Pneumonia"
    ][0]
    assert pneumonia.mapping_type.value == "approximate"
    assert not pneumonia.include_in_cross_dataset_metrics


def test_common_intersection_excludes_unsafe_or_missing_labels_with_reason() -> None:
    ontology = load_ontology(ONTOLOGY)

    result = ontology.safe_common_labels(["nih", "chexpert", "mimic", "vindr"])

    assert "Atelectasis" in result.included
    assert "Pleural Effusion" in result.included
    assert "Pneumonia" not in result.included
    assert "vindr" in result.excluded["Pneumonia"]
    assert "Lung Opacity" not in result.included
    assert result.excluded["Lung Opacity"]


def test_non_exact_mapping_cannot_enter_primary_metrics() -> None:
    with pytest.raises(ValueError, match="Only exact mappings"):
        LabelMapping.from_dict(
            {
                "canonical_label": "Pneumonia",
                "source_dataset": "rsna",
                "source_label": "Lung Opacity",
                "mapping_type": "approximate",
                "mapping_confidence": "low",
                "include_in_cross_dataset_metrics": True,
                "notes": "unsafe",
            }
        )
