"""
Base Field Extractor Interface (Phase 18 & 21)
Defines standard field extraction contract and transparent confidence scoring methodology.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseExtractor(ABC):
    def __init__(self, doc_type: str):
        self.doc_type = doc_type

    @abstractmethod
    def extract(self, text: str, entities: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extracts schema-specific fields with values and confidence ratings"""
        pass

    @staticmethod
    def wrap_field(value: Any, confidence: float, source_rule: str = "Regex/NER Match") -> Dict[str, Any]:
        """
        Standard field structure with transparent confidence calculation:
        - High: >= 0.90
        - Medium: 0.70 - 0.89
        - Low: < 0.70
        """
        conf_level = "High" if confidence >= 0.90 else ("Medium" if confidence >= 0.70 else "Low")
        return {
            "value": value,
            "confidence": round(confidence, 3),
            "level": conf_level,
            "extraction_method": source_rule
        }
