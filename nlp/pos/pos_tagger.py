"""
Part of Speech (POS) Tagging Module (Phase 7C)
Assigns syntactic tags (NOUN, VERB, ADJ, PROPN, NUM, etc.) to extracted tokens.
"""

from typing import List, Dict, Any, Tuple

class POSTagger:
    def __init__(self):
        # Lexicon of known grammatical markers
        self.tags = {
            'nouns': {'invoice', 'receipt', 'date', 'customer', 'amount', 'total', 'subtotal', 'tax', 'item', 'order', 'license', 'storage', 'support', 'coffee', 'muffin', 'sandwich', 'water', 'engineer', 'resume', 'skills', 'experience', 'education', 'card', 'identity', 'address', 'gender', 'dob'},
            'verbs': {'paid', 'pay', 'billed', 'order', 'ordered', 'due', 'developed', 'architected', 'deployed', 'implemented', 'approved', 'thank', 'learning', 'handling'},
            'adjectives': {'total', 'grand', 'senior', 'technical', 'large', 'mineral', 'professional', 'accomplished', 'clean', 'national', 'full', 'due', 'online', 'ready'},
            'proper_nouns': {'techflow', 'solutions', 'pvt', 'ltd', 'rahul', 'shah', 'gujarat', 'mg', 'road', 'bengaluru', 'ananya', 'verma', 'vikram', 'singh', 'delhi', 'india', 'iit', 'aroma', 'cafe'},
            'numbers': {'1', '2', '3', '4', '5', '10', '18', '24', '2026', '76500', '90270', '50000', '15000', '11500', '13770', '860', '903', '380', '180', '240', '60', '482', '12', '06', '1995'}
        }

    def tag_token(self, token: str) -> str:
        low = token.lower()
        if low.isdigit() or low.replace('.', '', 1).isdigit():
            return "NUM" # Number
        if token[0].isupper() and low in self.tags['proper_nouns']:
            return "PROPN" # Proper Noun
        if low in self.tags['verbs'] or low.endswith('ed') or low.endswith('ing'):
            return "VERB" # Verb
        if low in self.tags['nouns'] or low.endswith('tion') or low.endswith('ment') or low.endswith('er'):
            return "NOUN" # Noun
        if low in self.tags['adjectives'] or low.endswith('al') or low.endswith('ive') or low.endswith('ous'):
            return "ADJ" # Adjective
        return "NOUN"

    def tag(self, tokens: List[str]) -> List[Dict[str, str]]:
        tagged = []
        for t in tokens:
            tag = self.tag_token(t)
            tagged.append({
                "token": t,
                "pos": tag,
                "tag_description": self._get_tag_desc(tag)
            })
        return tagged

    def _get_tag_desc(self, tag: str) -> str:
        descriptions = {
            "NOUN": "Noun (common)",
            "PROPN": "Proper Noun (named entity)",
            "VERB": "Verb (action or state)",
            "ADJ": "Adjective (modifier)",
            "NUM": "Numeral (cardinal quantity)"
        }
        return descriptions.get(tag, "Word token")
