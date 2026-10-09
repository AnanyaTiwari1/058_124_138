"""
Full System Integration Tests
Validates End-to-End Processing, Schema Extraction, Multimodal Fusion, Chatbot, Summarizer, Database, Exports, REST API, and Full Syllabus Experiments.
"""

import os
import pytest
from fastapi.testclient import TestClient
from app.coordinator import DocumentIntelligenceCoordinator
from app.main import app
from deep_learning.object_detection.yolo_detector import DocumentRegionDetector
from experiments.nlpl.exp3_social_media_sentiment import run_social_media_sentiment_experiment
from experiments.nlpl.exp4_word2vec_glove_classifier import run_word_vector_classifier_experiment

client = TestClient(app)

@pytest.fixture
def coordinator():
    return DocumentIntelligenceCoordinator()

def test_invoice_full_processing(coordinator):
    sample_path = "data/sample_documents/invoice_sample.jpg"
    if os.path.exists(sample_path):
        record = coordinator.process_document(sample_path)
        assert record["document_type"] == "invoice"
        assert record["confidence"] > 0.85
        assert "invoice_number" in record["extracted_data"]["fields"]
        assert "grand_total" in record["extracted_data"]["fields"]
        assert record["validation_results"]["is_valid"] is True
        assert len(record["summary"]) > 20

def test_receipt_full_processing(coordinator):
    sample_path = "data/sample_documents/receipt_sample.jpg"
    if os.path.exists(sample_path):
        record = coordinator.process_document(sample_path)
        assert record["document_type"] == "receipt"
        assert "total_amount" in record["extracted_data"]["fields"]

def test_resume_full_processing(coordinator):
    sample_path = "data/sample_documents/resume_sample.jpg"
    if os.path.exists(sample_path):
        record = coordinator.process_document(sample_path)
        assert record["document_type"] == "resume"
        assert "skills" in record["extracted_data"]["fields"]

def test_id_card_full_processing(coordinator):
    sample_path = "data/sample_documents/id_card_sample.jpg"
    if os.path.exists(sample_path):
        record = coordinator.process_document(sample_path)
        assert record["document_type"] == "id_card"
        assert "identity_number" in record["extracted_data"]["fields"]

def test_yolo_region_detector():
    detector = DocumentRegionDetector()
    sample_path = "data/sample_documents/invoice_sample.jpg"
    if os.path.exists(sample_path):
        out_img = "data/processed/test_yolo.jpg"
        res = detector.detect_regions(sample_path, out_img)
        assert res["detection_count"] >= 3
        assert os.path.exists(out_img)

def test_social_media_sentiment_exp():
    res = run_social_media_sentiment_experiment()
    assert res["status"] == "COMPLETED"
    assert len(res["samples_analyzed"]) == 4

def test_word2vec_glove_classifier_exp():
    res = run_word_vector_classifier_experiment()
    assert res["status"] == "COMPLETED"
    assert res["word2vec_accuracy"] == "100%"

def test_chatbot_qa(coordinator):
    sample_data = {
        "fields": {
            "grand_total": {"value": 90270.0},
            "customer": {"value": "Rahul Shah"},
            "invoice_number": {"value": "INV-2026-1023"}
        }
    }
    ans1 = coordinator.chatbot.answer_question("What is the total amount?", sample_data)
    assert "90,270" in ans1["answer"]
    
    ans2 = coordinator.chatbot.answer_question("Who is the customer?", sample_data)
    assert "Rahul Shah" in ans2["answer"]

def test_exports(coordinator):
    sample_record = {
        "document_id": "TEST-DOC-001",
        "extracted_data": {
            "fields": {
                "total": {"value": 1000, "confidence": 0.95, "level": "High", "extraction_method": "Regex"}
            }
        }
    }
    json_p = coordinator.export_manager.export_json("TEST-DOC-001", sample_record)
    csv_p = coordinator.export_manager.export_csv("TEST-DOC-001", sample_record)
    xlsx_p = coordinator.export_manager.export_excel("TEST-DOC-001", sample_record)
    
    assert os.path.exists(json_p)
    assert os.path.exists(csv_p)
    assert os.path.exists(xlsx_p)

def test_fastapi_endpoints():
    res = client.get("/")
    assert res.status_code == 200
    assert "Multimodal Document Intelligence" in res.json()["platform"]
    
    res_dll = client.get("/api/experiments/dll")
    assert res_dll.status_code == 200
    assert "exp1_mlp_backprop" in res_dll.json()
    
    res_nlpl = client.get("/api/experiments/nlpl")
    assert res_nlpl.status_code == 200
    assert "exp1_bow_vs_tfidf" in res_nlpl.json()
