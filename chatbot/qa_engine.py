"""
Document Chatbot & Question Answering Engine (Phase 23)
Enables conversational interactions with uploaded documents using field lookups,
semantic context retrieval, and Groq LLM acceleration (Llama 3.3 / Mixtral).
"""

import os
import re
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

class DocumentChatbot:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            # Try reading from .env file if not in os.environ
            env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("GROQ_API_KEY="):
                            self.api_key = line.strip().split("=", 1)[1]
                        elif line.startswith("GROQ_MODEL=") and not model:
                            model = line.strip().split("=", 1)[1]
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    def _check_export_intent(self, question: str) -> Optional[Dict[str, str]]:
        """Detects if user is asking to export or download data in a specific format."""
        q = question.lower().strip()
        if any(w in q for w in ["export", "download", "save as", "generate file", "give me"]):
            if any(w in q for w in ["excel", "xlsx", "spreadsheet", "sheets"]):
                return {"format": "excel", "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "ext": "xlsx"}
            elif any(w in q for w in ["csv", "comma separated"]):
                return {"format": "csv", "mime": "text/csv", "ext": "csv"}
            elif any(w in q for w in ["json", "raw data"]):
                return {"format": "json", "mime": "application/json", "ext": "json"}
        return None

    def _query_groq_llm(self, question: str, extracted_data: Dict[str, Any], raw_text: str = "", history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Calls Groq Cloud API with multi-turn conversation memory for conversational reasoning."""
        if not self.api_key or self.api_key.startswith("gsk_your_"):
            return None
        
        system_prompt = (
            "You are an expert AI Document Intelligence assistant. You are given extracted structured data "
            "and raw OCR text from a document (such as an invoice, receipt, resume, or ID card). "
            "Answer the user's question accurately, concisely, and factually based strictly on the provided document context. "
            "Maintain conversational context from previous messages if applicable. "
            "If currency is involved, format it cleanly. If information is not in the document, state that clearly."
        )
        
        fields_summary = {}
        fields = extracted_data.get("fields", extracted_data)
        for k, v in fields.items():
            if isinstance(v, dict) and "value" in v:
                fields_summary[k] = v["value"]
            else:
                fields_summary[k] = v

        doc_context = (
            f"--- DOCUMENT STRUCTURED DATA ---\n"
            f"{json.dumps(fields_summary, indent=2, default=str)}\n\n"
            f"--- RAW OCR TEXT (IF ANY) ---\n"
            f"{raw_text[:1500] if raw_text else 'N/A'}"
        )

        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{doc_context}"}
        ]

        # Add multi-turn conversation history (up to last 6 turns)
        if history:
            for turn in history[-6:]:
                role = "user" if turn.get("role") == "user" else "assistant"
                content = turn.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": question})

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
            "User-Agent": "MultimodalDocIntelligence/1.0"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 450
        }

        try:
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                reply = res_data["choices"][0]["message"]["content"].strip()
                return {
                    "answer": reply,
                    "confidence": 0.99,
                    "source_field": f"groq_llm ({self.model})"
                }
        except Exception as e:
            return None

    def answer_question(self, question: str, extracted_data: Dict[str, Any], raw_text: str = "", history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Answers natural language queries using Groq LLM with deterministic heuristic fallback,
        multi-turn context awareness, and automated export detection.
        """
        # 1. Check if question is an in-chat export command
        export_info = self._check_export_intent(question)
        if export_info:
            fmt = export_info["format"].upper()
            return {
                "answer": f"Certainly! I have prepared the **{fmt}** export for this document. You can download it directly below.",
                "confidence": 1.0,
                "source_field": "export_intent_handler",
                "export_intent": export_info
            }

        # 2. Attempt high-intelligence Groq LLM query with conversation history
        groq_result = self._query_groq_llm(question, extracted_data, raw_text, history=history)
        if groq_result:
            return groq_result

        q = question.lower().strip()
        fields = extracted_data.get("fields", extracted_data)
        
        # 1. Total / Amount queries
        if any(w in q for w in ["total", "how much", "cost", "grand total", "bill amount", "pay"]):
            total_val = fields.get("grand_total", {}).get("value") or fields.get("total_amount", {}).get("value")
            if total_val:
                return {
                    "answer": f"The total amount is ₹{total_val:,.2f}." if isinstance(total_val, (int, float)) else f"The total amount is {total_val}.",
                    "confidence": 0.98,
                    "source_field": "grand_total"
                }

        # 2. Most expensive item queries
        if "most expensive" in q or "highest" in q or "costliest" in q:
            items = fields.get("items", {}).get("value", [])
            if items:
                # Find item with max price/amount
                best_item = None
                max_amt = -1
                for it in items:
                    amt = it.get("amount", it.get("unit_price", 0))
                    if amt > max_amt:
                        max_amt = amt
                        best_item = it.get("description", it.get("item", "Item"))
                if best_item:
                    return {
                        "answer": f"The most expensive item is '{best_item}' priced at ₹{max_amt:,.2f}.",
                        "confidence": 0.95,
                        "source_field": "items"
                    }

        # 3. Customer / Person / Candidate Name queries
        if any(w in q for w in ["customer", "who", "name", "client", "billed to", "candidate"]):
            person = fields.get("customer", {}).get("value") or fields.get("full_name", {}).get("value")
            if person:
                return {
                    "answer": f"The customer/person identified on this document is {person}.",
                    "confidence": 0.96,
                    "source_field": "customer / full_name"
                }

        # 4. Invoice Number / ID queries
        if any(w in q for w in ["invoice number", "invoice #", "id number", "order number", "order #"]):
            inv_no = fields.get("invoice_number", {}).get("value") or fields.get("order_number", {}).get("value") or fields.get("identity_number", {}).get("value")
            if inv_no:
                return {
                    "answer": f"The identification number on this document is {inv_no}.",
                    "confidence": 0.98,
                    "source_field": "invoice_number / id_number"
                }

        # 5. Date queries
        if "date" in q or "when" in q or "dob" in q:
            date_val = fields.get("invoice_date", {}).get("value") or fields.get("date", {}).get("value") or fields.get("date_of_birth", {}).get("value")
            if date_val:
                return {
                    "answer": f"The date recorded on this document is {date_val}.",
                    "confidence": 0.97,
                    "source_field": "date"
                }

        # 6. GSTIN / Tax queries
        if "gstin" in q or "gst" in q or "tax" in q:
            gstin = fields.get("gstin", {}).get("value")
            tax_val = fields.get("tax", {}).get("value")
            if gstin or tax_val:
                ans = []
                if gstin: ans.append(f"GSTIN is {gstin}")
                if tax_val: ans.append(f"Tax amount is ₹{tax_val:,.2f}" if isinstance(tax_val, (int, float)) else f"Tax is {tax_val}")
                return {
                    "answer": " and ".join(ans) + ".",
                    "confidence": 0.94,
                    "source_field": "gstin / tax"
                }

        # 7. Skills & Experience (Resume)
        if "skill" in q or "technolog" in q:
            skills = fields.get("skills", {}).get("value", [])
            if skills:
                return {
                    "answer": f"Key technical skills listed include: {', '.join(skills[:8])}.",
                    "confidence": 0.95,
                    "source_field": "skills"
                }

        # Fallback: General keyword scan over raw text
        return {
            "answer": "I found the document context, but no specific value directly matched your question. Please check the extracted fields or rephrase your question.",
            "confidence": 0.50,
            "source_field": "fallback"
        }
