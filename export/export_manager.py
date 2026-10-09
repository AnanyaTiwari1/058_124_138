"""
Export Manager Module (Phase 26)
Provides export services for structured document intelligence outputs into JSON, CSV, and Excel formats.
"""

import os
import json
import pandas as pd
from typing import Dict, Any

class ExportManager:
    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def export_json(self, doc_id: str, data: Dict[str, Any]) -> str:
        filepath = os.path.join(self.output_dir, f"{doc_id}_export.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath

    def export_csv(self, doc_id: str, data: Dict[str, Any]) -> str:
        filepath = os.path.join(self.output_dir, f"{doc_id}_export.csv")
        fields = data.get("extracted_data", {}).get("fields", data.get("extracted_data", {}))
        
        flat_records = []
        for key, info in fields.items():
            if isinstance(info, dict) and "value" in info:
                val = info["value"]
                if isinstance(val, list):
                    val = json.dumps(val)
                flat_records.append({
                    "Field": key,
                    "Value": str(val),
                    "Confidence": info.get("confidence", 0.0),
                    "Level": info.get("level", "N/A"),
                    "Extraction Method": info.get("extraction_method", "N/A")
                })
        
        df = pd.DataFrame(flat_records)
        df.to_csv(filepath, index=False)
        return filepath

    def export_excel(self, doc_id: str, data: Dict[str, Any]) -> str:
        filepath = os.path.join(self.output_dir, f"{doc_id}_export.xlsx")
        fields = data.get("extracted_data", {}).get("fields", data.get("extracted_data", {}))
        
        flat_records = []
        for key, info in fields.items():
            if isinstance(info, dict) and "value" in info:
                val = info["value"]
                if isinstance(val, list):
                    val = json.dumps(val)
                flat_records.append({
                    "Field": key,
                    "Value": str(val),
                    "Confidence": info.get("confidence", 0.0),
                    "Level": info.get("level", "N/A"),
                    "Extraction Method": info.get("extraction_method", "N/A")
                })
                
        df = pd.DataFrame(flat_records)
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Extracted Fields', index=False)
            
        return filepath
