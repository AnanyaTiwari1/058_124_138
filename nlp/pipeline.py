"""
NLPL Part 1 Comprehensive Pipeline
Coordinates Text Extraction, Cleaning, BoW, TF-IDF, N-Grams, NER, and POS.
"""

from typing import Dict, Any, List
from nlp.preprocessing.cleaner import TextCleaner
from nlp.vectorization.bow_tfidf import Vectorizer
from nlp.ngrams.ngram_analyzer import NGramAnalyzer
from nlp.ner.ner_extractor import NERExtractor
from nlp.pos.pos_tagger import POSTagger

class NLPLPipelinePart1:
    def __init__(self, corpus: List[str] = None):
        self.cleaner = TextCleaner()
        self.vectorizer = Vectorizer(corpus)
        self.ngram_analyzer = NGramAnalyzer()
        self.ner_extractor = NERExtractor()
        self.pos_tagger = POSTagger()

    def process_document_text(self, raw_text: str) -> Dict[str, Any]:
        """
        Executes full NLPL Part 1 Pipeline on extracted document text.
        """
        # 1. Text Preprocessing
        clean_result = self.cleaner.process(raw_text)
        cleaned_tokens = clean_result["cleaned_tokens"]
        
        # 2. BoW & TF-IDF
        bow_result = self.vectorizer.compute_bow(cleaned_tokens)
        tfidf_result = self.vectorizer.compute_tfidf(cleaned_tokens)
        bow_vs_tfidf = self.vectorizer.compare_bow_vs_tfidf(cleaned_tokens)
        
        # 3. N-Grams
        ngrams_result = self.ngram_analyzer.analyze(cleaned_tokens)
        
        # 4. NER
        entities = self.ner_extractor.extract_entities(raw_text)
        
        # 5. POS Tagging
        pos_tags = self.pos_tagger.tag(cleaned_tokens[:40])
        
        return {
            "preprocessing": clean_result,
            "vectorization": {
                "bow": bow_result,
                "tfidf": tfidf_result,
                "comparison": bow_vs_tfidf
            },
            "ngrams": ngrams_result,
            "ner": entities,
            "pos": pos_tags
        }
