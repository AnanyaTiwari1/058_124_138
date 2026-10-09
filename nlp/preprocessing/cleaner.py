"""
NLP Preprocessing Module (Phase 4)
Implements text tokenization, stop-word removal, punctuation removal, stemming, and lemmatization.
Preserves both raw text and processed text.
"""

import re
import string
from typing import List, Dict, Any

class TextCleaner:
    def __init__(self):
        # Comprehensive English stop words list
        self.stop_words = {
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 
            'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 
            'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t', 
            'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 
            'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 
            'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 
            'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 
            'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 
            'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 
            'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 
            'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 
            'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 
            'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 
            'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 
            'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 
            'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 
            'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 
            'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 
            'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 
            'yourself', 'yourselves'
        }

    def tokenize(self, text: str) -> List[str]:
        """Word tokenization splitting on word boundaries"""
        return re.findall(r'\b\w+\b', text)

    def remove_punctuation(self, text: str) -> str:
        """Removes punctuation characters while retaining alphanumeric terms and whitespace"""
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filters out standard English stop words"""
        return [t for t in tokens if t.lower() not in self.stop_words]

    def stem_word(self, word: str) -> str:
        """Porter Stemmer implementation for morphological reduction"""
        w = word.lower()
        # Common suffix stripping rules
        if w.endswith('sses'): w = w[:-2]
        elif w.endswith('ies'): w = w[:-2]
        elif w.endswith('ss'): pass
        elif w.endswith('s'): w = w[:-1]
        
        if w.endswith('eed'):
            if len(w) > 4: w = w[:-1]
        elif w.endswith('ing') and len(w) > 4:
            w = w[:-3]
        elif w.endswith('ed') and len(w) > 3:
            w = w[:-2]
            
        if w.endswith('ational'): w = w[:-5] + 'e'
        elif w.endswith('tional'): w = w[:-4]
        elif w.endswith('izer'): w = w[:-1]
        elif w.endswith('ation'): w = w[:-3] + 'e'
        elif w.endswith('ator'): w = w[:-2] + 'e'
        elif w.endswith('alism'): w = w[:-3]
        elif w.endswith('ness'): w = w[:-4]
        elif w.endswith('ment'): w = w[:-4]
        elif w.endswith('able'): w = w[:-4]
        return w

    def lemmatize_word(self, word: str) -> str:
        """Rule-based lemmatization to base dictionary forms"""
        w = word.lower()
        irregular_lemmas = {
            'bought': 'buy', 'paid': 'pay', 'due': 'due', 'services': 'service',
            'invoices': 'invoice', 'receipts': 'receipt', 'taxes': 'tax',
            'solutions': 'solution', 'billed': 'bill', 'ordered': 'order',
            'charges': 'charge', 'items': 'item', 'details': 'detail',
            'technologies': 'technology', 'engineers': 'engineer'
        }
        if w in irregular_lemmas:
            return irregular_lemmas[w]
        return self.stem_word(w)

    def process(self, raw_text: str) -> Dict[str, Any]:
        """
        Executes complete preprocessing pipeline while preserving raw text.
        """
        tokens = self.tokenize(raw_text)
        cleaned_tokens = self.remove_stopwords(tokens)
        stemmed_tokens = [self.stem_word(t) for t in cleaned_tokens]
        lemmatized_tokens = [self.lemmatize_word(t) for t in cleaned_tokens]
        
        return {
            "raw_text": raw_text,
            "raw_tokens": tokens,
            "cleaned_tokens": cleaned_tokens,
            "stemmed_tokens": stemmed_tokens,
            "lemmatized_tokens": lemmatized_tokens,
            "cleaned_text": " ".join(cleaned_tokens),
            "stats": {
                "raw_token_count": len(tokens),
                "cleaned_token_count": len(cleaned_tokens),
                "stopwords_removed": len(tokens) - len(cleaned_tokens)
            }
        }
