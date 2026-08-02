"""Canonical chest X-ray label ontology for Academic V2."""

from medguard.ontology.mapper import LabelOntology, load_ontology
from medguard.ontology.schema import LabelMapping, MappingType

__all__ = ["LabelMapping", "LabelOntology", "MappingType", "load_ontology"]
