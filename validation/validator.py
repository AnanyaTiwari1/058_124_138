"""
Document Validation Engine (Phase 20)
Performs rule-based, mathematical, and schema validations on extracted fields.
"""

import re
from typing import Dict, Any, List

class DocumentValidator:
    @staticmethod
    def validate_document(doc_type: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates extracted fields according to document type rules.
        """
        checks = []
        warnings = []
        is_valid = True

        if doc_type == "invoice":
            # 1. Invoice Number Check
            inv_no = fields.get("invoice_number", {}).get("value")
            if inv_no:
                checks.append({"rule": "Invoice Number Present", "status": "PASSED", "detail": f"ID: {inv_no}"})
            else:
                checks.append({"rule": "Invoice Number Present", "status": "FAILED", "detail": "Missing invoice number"})
                is_valid = False

            # 2. Mathematical Consistency Check (Subtotal + Tax == Total)
            subtotal = float(fields.get("subtotal", {}).get("value", 0))
            tax = float(fields.get("tax", {}).get("value", 0))
            grand_total = float(fields.get("grand_total", {}).get("value", 0))
            
            calculated_total = round(subtotal + tax, 2)
            if abs(calculated_total - grand_total) < 1.0:
                checks.append({"rule": "Arithmetic Validation", "status": "PASSED", "detail": f"Subtotal ({subtotal}) + Tax ({tax}) == Total ({grand_total})"})
            else:
                checks.append({"rule": "Arithmetic Validation", "status": "WARNING", "detail": f"Mismatch: {subtotal} + {tax} != {grand_total}"})
                warnings.append("Grand total does not exactly match sum of subtotal and tax.")

            # 3. GSTIN Check
            gstin = fields.get("gstin", {}).get("value", "")
            if re.match(r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$', gstin):
                checks.append({"rule": "GSTIN Structure Check", "status": "PASSED", "detail": f"Valid 15-char format: {gstin}"})
            else:
                checks.append({"rule": "GSTIN Structure Check", "status": "WARNING", "detail": "GSTIN format irregular"})
                warnings.append("GSTIN structure format irregular.")

        elif doc_type == "receipt":
            total = fields.get("total_amount", {}).get("value")
            if total:
                checks.append({"rule": "Receipt Total Present", "status": "PASSED", "detail": f"Amount: {total}"})
            else:
                checks.append({"rule": "Receipt Total Present", "status": "FAILED", "detail": "Total missing"})
                is_valid = False

        elif doc_type == "resume":
            email = fields.get("email", {}).get("value", "")
            if "@" in email and "." in email:
                checks.append({"rule": "Email Format", "status": "PASSED", "detail": email})
            else:
                checks.append({"rule": "Email Format", "status": "FAILED", "detail": "Invalid or missing email"})
                is_valid = False

            skills = fields.get("skills", {}).get("value", [])
            if len(skills) >= 3:
                checks.append({"rule": "Skills Inventory Check", "status": "PASSED", "detail": f"{len(skills)} skills detected"})
            else:
                checks.append({"rule": "Skills Inventory Check", "status": "WARNING", "detail": "Few skills extracted"})

        elif doc_type == "id_card":
            id_no = fields.get("identity_number", {}).get("value", "")
            if id_no:
                checks.append({"rule": "National ID Number Check", "status": "PASSED", "detail": id_no})
            else:
                checks.append({"rule": "National ID Number Check", "status": "FAILED", "detail": "ID number missing"})
                is_valid = False

        passed_count = sum(1 for c in checks if c["status"] == "PASSED")
        total_checks = len(checks) if checks else 1
        quality_score = round((passed_count / total_checks) * 100, 1)

        return {
            "is_valid": is_valid,
            "quality_score": quality_score,
            "checks": checks,
            "warnings": warnings,
            "status_badge": "VERIFIED" if is_valid and quality_score >= 80 else "NEEDS_REVIEW"
        }
