"""
FastAPI REST API Service for Multimodal Document Intelligence Platform (Phase 25)
Exposes end-to-end processing, field extraction, chatbot Q&A, and syllabus experiment endpoints.
"""

import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.coordinator import DocumentIntelligenceCoordinator
from experiments.dll.exp1_mlp_backprop import run_mlp_backprop_experiment
from experiments.dll.exp2_mnist import run_mnist_experiment
from experiments.dll.exp3_autoencoder import run_autoencoder_experiment
from experiments.dll.exp4_vae import run_vae_experiment
from experiments.dll.exp5_cae import run_cae_experiment
from experiments.dll.exp6_gan import run_gan_experiment
from experiments.dll.exp7_sentiment_rnn import run_sentiment_rnn_experiment
from experiments.nlpl.exp1_bow_tfidf_comparison import run_bow_tfidf_experiment
from experiments.nlpl.exp2_ngram_modelling import run_ngram_experiment
from deep_learning.transfer_learning.transfer_model import TransferLearningEvaluator

app = FastAPI(
    title="Multimodal Document Intelligence API",
    description="Unified REST API serving NLPL and DLL academic workflows and production document extraction.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

coordinator = DocumentIntelligenceCoordinator()
transfer_evaluator = TransferLearningEvaluator()

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {
        "platform": "Multimodal Document Intelligence Platform",
        "academic_projects": [
            "Project 1 (NLPL): Intelligent Document Information Extraction and Conversational Analysis using NLP",
            "Project 2 (DLL): Deep Learning Based Multimodal Document Classification and Understanding System"
        ],
        "status": "ONLINE",
        "docs_url": "/docs"
    }

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    try:
        saved_info = coordinator.doc_manager.save_upload(file.filename, content)
        # Automatically trigger end-to-end multimodal processing
        record = coordinator.process_document(saved_info["file_path"], file.filename)
        return record
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/documents")
def list_documents():
    return coordinator.db.list_documents(limit=50)

@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    doc = coordinator.db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@app.post("/api/documents/{doc_id}/chat")
def chat_with_document(doc_id: str, req: ChatRequest):
    doc = coordinator.db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    extracted_data = doc.get("extracted_data", {})
    return coordinator.chatbot.answer_question(req.question, extracted_data)

@app.get("/api/documents/{doc_id}/summary")
def get_document_summary(doc_id: str):
    doc = coordinator.db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document_id": doc_id, "summary": doc.get("summary", "")}

@app.get("/api/documents/{doc_id}/export/{export_format}")
def export_document(doc_id: str, export_format: str):
    doc = coordinator.db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    fmt = export_format.lower()
    if fmt == "json":
        path = coordinator.export_manager.export_json(doc_id, doc)
    elif fmt == "csv":
        path = coordinator.export_manager.export_csv(doc_id, doc)
    elif fmt in ["excel", "xlsx"]:
        path = coordinator.export_manager.export_excel(doc_id, doc)
    else:
        raise HTTPException(status_code=400, detail="Supported export formats: json, csv, excel")
        
    return {"document_id": doc_id, "format": fmt, "exported_file": path}

# Academic Syllabus Experiment Endpoints
@app.get("/api/experiments/nlpl")
def get_nlpl_experiments():
    return {
        "exp1_bow_vs_tfidf": run_bow_tfidf_experiment(),
        "exp2_ngrams": run_ngram_experiment()
    }

@app.get("/api/experiments/dll")
def get_dll_experiments():
    return {
        "exp1_mlp_backprop": run_mlp_backprop_experiment(epochs=15),
        "exp2_mnist": run_mnist_experiment(7),
        "exp3_autoencoder": run_autoencoder_experiment(),
        "exp4_vae": run_vae_experiment(),
        "exp5_cae": run_cae_experiment(),
        "exp6_gan": run_gan_experiment(),
        "exp7_sentiment_rnn": run_sentiment_rnn_experiment(),
        "transfer_learning_comparison": transfer_evaluator.compare_vgg_vs_resnet()
    }
