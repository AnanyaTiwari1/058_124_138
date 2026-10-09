"""
Unit tests for NLPL Part 1 Pipeline
"""

import os
import pytest
from nlp.preprocessing.cleaner import TextCleaner
from nlp.vectorization.bow_tfidf import Vectorizer
from nlp.ngrams.ngram_analyzer import NGramAnalyzer
from nlp.ner.ner_extractor import NERExtractor
from nlp.pos.pos_tagger import POSTagger
from nlp.pipeline import NLPLPipelinePart1
from document.text_extractor import TextExtractor

SAMPLE_INVOICE_TEXT = """
TECHFLOW SOLUTIONS PVT LTD TAX INVOICE
Invoice Number: INV-2026-1023
Date: 15 August 2026
Seller: TechFlow Solutions Pvt Ltd
GSTIN: 24ABCDE1234F1Z5
Billed To: Rahul Shah
Email: rahul.shah@example.com | Phone: +91-9876543210
1 Enterprise AI Server License Qty: 1 Unit Price: INR 50000.00 Amount: INR 50000.00
Subtotal: INR 76500.00
Tax / GST (18%): INR 13770.00
Grand Total: INR 90270.00
"""

def test_text_cleaner():
    cleaner = TextCleaner()
    res = cleaner.process("The invoice was paid by Rahul Shah.")
    assert "invoice" in res["cleaned_tokens"]
    assert "paid" in res["cleaned_tokens"]
    assert "the" not in res["cleaned_tokens"]
    assert len(res["stemmed_tokens"]) == len(res["cleaned_tokens"])
    assert len(res["lemmatized_tokens"]) == len(res["cleaned_tokens"])

def test_vectorizer():
    vectorizer = Vectorizer([SAMPLE_INVOICE_TEXT])
    tokens = ["invoice", "tax", "license", "total", "invoice"]
    bow = vectorizer.compute_bow(tokens)
    assert bow["token_counts"]["invoice"] == 2
    tfidf = vectorizer.compute_tfidf(tokens)
    assert "invoice" in tfidf["tfidf_scores"]
    comp = vectorizer.compare_bow_vs_tfidf(tokens)
    assert len(comp) > 0

def test_ngrams():
    analyzer = NGramAnalyzer()
    tokens = ["total", "amount", "due", "on", "receipt"]
    res = analyzer.analyze(tokens)
    assert any(item["ngram"] == "total amount" for item in res["bigrams"])
    assert any(item["ngram"] == "total amount due" for item in res["trigrams"])

def test_ner_extractor():
    ner = NERExtractor()
    entities = ner.extract_entities(SAMPLE_INVOICE_TEXT)
    entity_types = {e["label"] for e in entities}
    assert "GSTIN" in entity_types
    assert "EMAIL" in entity_types
    assert "MONEY" in entity_types

def test_pos_tagger():
    tagger = POSTagger()
    tags = tagger.tag(["Rahul", "paid", "invoice", "90270"])
    pos_map = {t["token"]: t["pos"] for t in tags}
    assert pos_map["90270"] == "NUM"

def test_full_nlp_pipeline():
    pipeline = NLPLPipelinePart1([SAMPLE_INVOICE_TEXT])
    result = pipeline.process_document_text(SAMPLE_INVOICE_TEXT)
    assert "preprocessing" in result
    assert "vectorization" in result
    assert "ngrams" in result
    assert "ner" in result
    assert "pos" in result
