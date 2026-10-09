# Multimodal Document Intelligence Platform
> **Dual Academic Projects in One Unified Platform**
> 
> - **Project 1 (NLPL):** *Intelligent Document Information Extraction and Conversational Analysis using Natural Language Processing*
> - **Project 2 (DLL):** *Deep Learning Based Multimodal Document Classification and Understanding System*

---

## 📖 Overview
The **Multimodal Document Intelligence Platform** is an academic and production-ready system integrating Natural Language Processing (NLPL) and Deep Learning (DLL) into a single cohesive document understanding pipeline.

```text
Document (PDF/Image)
       │
       ▼
Pre-processing & Enhancement
       │
       ▼
Text Extraction (OCR)
       │
       ├──► NLPL Part 1: Tokenization, Stopwords, Stemming/Lemmatization, BoW, TF-IDF, N-Grams, NER, POS
       │
       ├──► DLL Core Engine: CNN, Transfer Learning (VGG/ResNet), LSTM, Autoencoders, GAN, MLP/Backprop
       │
       └──► NLPL Part 2: Word2Vec/GloVe, BERT, Multimodal Fusion, Extraction, Validation, Chatbot, Summary
       │
       ▼
Structured JSON ──► Database (MongoDB) ──► Chat / Summary / Export (JSON, CSV, Excel)
```

---

## 🚀 Getting Started

### 1. Requirements
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Interactive Platform UI
```bash
streamlit run frontend/app.py
```

### 4. Run the REST API
```bash
uvicorn app.main:app --reload --port 8000
```
API Documentation will be available at `http://localhost:8000/docs`.

### 5. Run Test Suite
```bash
pytest tests/
```
