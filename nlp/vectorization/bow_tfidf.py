"""
Bag of Words (BoW) & TF-IDF Module (Phase 5)
Calculates term frequencies, inverse document frequencies, and side-by-side BoW vs. TF-IDF comparisons.
"""

import math
from collections import Counter
from typing import List, Dict, Any, Tuple

class Vectorizer:
    def __init__(self, corpus: List[str] = None):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count: int = 0
        if corpus:
            self.fit(corpus)

    def fit(self, corpus: List[str]):
        """Fits vocabulary and computes IDF scores over corpus"""
        self.doc_count = len(corpus)
        all_words = set()
        doc_freqs = Counter()

        for doc in corpus:
            tokens = [t.lower() for t in doc.split()]
            unique_in_doc = set(tokens)
            all_words.update(unique_in_doc)
            for w in unique_in_doc:
                doc_freqs[w] += 1

        sorted_vocab = sorted(list(all_words))
        self.vocabulary = {word: idx for idx, word in enumerate(sorted_vocab)}
        
        # Smooth IDF: ln((1 + N) / (1 + df)) + 1
        for word, idx in self.vocabulary.items():
            df = doc_freqs[word]
            self.idf[word] = math.log((1 + self.doc_count) / (1 + df)) + 1.0

    def compute_bow(self, tokens: List[str]) -> Dict[str, Any]:
        """Computes Bag of Words frequency counts and vector representation"""
        lowered = [t.lower() for t in tokens]
        counts = Counter(lowered)
        
        vector = [0] * len(self.vocabulary)
        for word, count in counts.items():
            if word in self.vocabulary:
                vector[self.vocabulary[word]] = count
                
        return {
            "token_counts": dict(counts.most_common(20)),
            "vocabulary_size": len(self.vocabulary) if self.vocabulary else len(counts),
            "vector": vector[:50] # truncated top elements for clean preview
        }

    def compute_tfidf(self, tokens: List[str]) -> Dict[str, Any]:
        """Computes TF-IDF weights and top distinctive keywords"""
        lowered = [t.lower() for t in tokens]
        total_tokens = len(lowered) if len(lowered) > 0 else 1
        tf_counts = Counter(lowered)
        
        tfidf_scores = {}
        for word, count in tf_counts.items():
            tf = count / total_tokens
            idf_val = self.idf.get(word, 1.5) # default IDF if unseen
            tfidf_scores[word] = round(tf * idf_val, 4)

        sorted_scores = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "tfidf_scores": dict(sorted_scores[:20]),
            "top_distinctive_terms": [item[0] for item in sorted_scores[:8]]
        }

    def compare_bow_vs_tfidf(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Academic Comparison Table: BoW (Raw Frequency) vs. TF-IDF (Distinctiveness Weight)
        """
        bow = self.compute_bow(tokens)["token_counts"]
        tfidf = self.compute_tfidf(tokens)["tfidf_scores"]
        
        comparison = []
        all_terms = list(dict.fromkeys(list(bow.keys()) + list(tfidf.keys())))[:15]
        
        for term in all_terms:
            freq = bow.get(term, 0)
            weight = tfidf.get(term, 0.0)
            comparison.append({
                "term": term,
                "bow_count": freq,
                "tfidf_weight": weight,
                "explanation": "High frequency common word" if freq > 3 and weight < 0.05 else "Distinctive key term"
            })
            
        return comparison
