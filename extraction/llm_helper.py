"""
LLM Schema Extraction Helper (Phase 18 / 23)
Leverages Groq LLM (openai/gpt-oss-120b) for zero-shot schema extraction on custom uploaded documents.
"""

import os
import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

def _get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key.startswith("gsk_your_"):
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("GROQ_API_KEY="):
                        key = line.strip().split("=", 1)[1]
                        break
    return key

def extract_schema_with_llm(doc_type: str, raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Calls Groq LLM to extract structured fields from raw OCR/PDF text.
    Returns parsed dictionary or None if unavailable.
    """
    api_key = _get_api_key()
    if not api_key or api_key.startswith("gsk_your_") or len(raw_text.strip()) < 10:
        return None

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    schema_prompts = {
        "invoice": (
            "Extract structured invoice fields in JSON format with these exact keys:\n"
            "- invoice_number (string)\n"
            "- invoice_date (string)\n"
            "- seller (string)\n"
            "- customer (string)\n"
            "- gstin (string or null)\n"
            "- items (list of objects with 'description', 'qty', 'unit_price', 'amount')\n"
            "- subtotal (number)\n"
            "- tax (number)\n"
            "- grand_total (number)\n"
        ),
        "receipt": (
            "Extract structured receipt fields in JSON format with these exact keys:\n"
            "- merchant (string)\n"
            "- order_number (string)\n"
            "- date (string)\n"
            "- items (list of objects with 'item', 'qty', 'amount')\n"
            "- subtotal (number)\n"
            "- tax (number)\n"
            "- total_amount (number)\n"
            "- payment_method (string)\n"
        ),
        "resume": (
            "Extract structured resume fields in JSON format with these exact keys:\n"
            "- full_name (string)\n"
            "- professional_title (string)\n"
            "- email (string)\n"
            "- phone (string)\n"
            "- location (string)\n"
            "- skills (list of strings)\n"
            "- work_experience (list of objects with 'role', 'company', 'duration', 'highlight')\n"
            "- education (list of objects with 'degree', 'institution', 'grade', 'year')\n"
        ),
        "id_card": (
            "Extract structured identity card fields in JSON format with these exact keys:\n"
            "- card_type (string)\n"
            "- full_name (string)\n"
            "- date_of_birth (string)\n"
            "- gender (string)\n"
            "- identity_number (string)\n"
            "- address (string)\n"
        )
    }

    instructions = schema_prompts.get(doc_type, schema_prompts["invoice"])
    
    prompt = (
        f"You are a high-precision document extraction engine.\n"
        f"{instructions}\n"
        f"Document text:\n\"\"\"\n{raw_text[:3500]}\n\"\"\"\n\n"
        f"Respond ONLY with a valid JSON object matching the requested keys. Do not include markdown fences or explanation."
    )

    try:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.05
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(req, timeout=8) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"].strip()
            # Clean possible markdown wrap
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            parsed = json.loads(content.strip())
            return parsed
    except Exception:
        return None
