"""
N-Grams Analysis Module (Phase 6)
Extracts Unigrams, Bigrams, and Trigrams for document phrase detection and statistical language modeling.
"""

from collections import Counter
from typing import List, Dict, Any, Tuple

class NGramAnalyzer:
    @staticmethod
    def generate_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Generates n-tuples of consecutive tokens"""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def analyze(self, tokens: List[str], top_k: int = 10) -> Dict[str, Any]:
        """
        Extracts top Unigrams, Bigrams, and Trigrams with frequencies.
        """
        unigrams = self.generate_ngrams(tokens, 1)
        bigrams = self.generate_ngrams(tokens, 2)
        trigrams = self.generate_ngrams(tokens, 3)

        unigram_counts = Counter([" ".join(g) for g in unigrams])
        bigram_counts = Counter([" ".join(g) for g in bigrams])
        trigram_counts = Counter([" ".join(g) for g in trigrams])

        return {
            "unigrams": [{"ngram": k, "frequency": v} for k, v in unigram_counts.most_common(top_k)],
            "bigrams": [{"ngram": k, "frequency": v} for k, v in bigram_counts.most_common(top_k)],
            "trigrams": [{"ngram": k, "frequency": v} for k, v in trigram_counts.most_common(top_k)],
            "totals": {
                "total_unigrams": len(unigrams),
                "total_bigrams": len(bigrams),
                "total_trigrams": len(trigrams)
            }
        }
