"""
Multimodal Document Intelligence Platform Coordinator
Main orchestration service connecting NLPL Part 1, DLL Engine, NLPL Part 2, and Application Storage.
"""

import os
from typing import Dict, Any, Optional

from document.upload import DocumentManager
from document.image_enhancer import ImageEnhancer, PDFProcessor
from document.text_extractor import TextExtractor
from nlp.pipeline import NLPLPipelinePart1
from deep_learning.cnn.document_classifier import DocumentImageClassifier
from deep_learning.lstm.sequence_classifier import SequenceAndSentimentAnalyzer
from fusion.multimodal_fusion import MultimodalFusionEngine
from extraction.invoice import InvoiceExtractor
from extraction.receipt import ReceiptExtractor
from extraction.resume import ResumeExtractor
from extraction.id_document import IDDocumentExtractor
from validation.validator import DocumentValidator
from summarization.summarizer import DocumentSummarizer
from database.db import DatabaseManager
from chatbot.qa_engine import DocumentChatbot
from export.export_manager import ExportManager

class DocumentIntelligenceCoordinator:
    def __init__(self):
        self.doc_manager = DocumentManager()
        self.text_extractor = TextExtractor()
        self.nlp_pipeline = NLPLPipelinePart1()
        self.cnn_classifier = DocumentImageClassifier()
        self.lstm_analyzer = SequenceAndSentimentAnalyzer()
        self.fusion_engine = MultimodalFusionEngine()
        self.validator = DocumentValidator()
        self.summarizer = DocumentSummarizer()
        self.db = DatabaseManager()
        self.chatbot = DocumentChatbot()
        self.export_manager = ExportManager()
        
        self.extractors = {
            "invoice": InvoiceExtractor(),
            "receipt": ReceiptExtractor(),
            "resume": ResumeExtractor(),
            "id_card": IDDocumentExtractor()
        }

    def process_document(self, file_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes full multimodal document understanding pipeline:
        Upload -> Enhance -> OCR -> NLPL 1 -> DLL Engine -> Fusion -> Extraction -> Validation -> Summary -> DB
        """
        if filename is None:
            filename = os.path.basename(file_path)

        # 1. Image Enhancement & Preprocessing
        enhanced_path, _ = ImageEnhancer.enhance_image(file_path)

        # 2. Text Extraction (OCR)
        ocr_result = self.text_extractor.extract_text(file_path)
        raw_text = ocr_result["raw_text"]

        # 3. NLPL Part 1 Pipeline
        nlp_res = self.nlp_pipeline.process_document_text(raw_text)

        # 4. DLL Visual Classification (CNN)
        visual_pred = self.cnn_classifier.predict(file_path)

        # 5. Multimodal Information Fusion
        fusion_result = self.fusion_engine.fuse(
            visual_pred=visual_pred,
            textual_tokens=nlp_res["preprocessing"]["cleaned_tokens"],
            extracted_entities=nlp_res["ner"]
        )
        doc_type = fusion_result["final_document_type"]

        # 6. Schema-Specific Field Extraction
        extractor = self.extractors.get(doc_type, self.extractors["invoice"])
        extracted_fields = extractor.extract(raw_text, nlp_res["ner"])

        # 7. Validation Rule Engine
        val_result = self.validator.validate_document(doc_type, extracted_fields)

        # 8. Document Summarization
        summary_text = self.summarizer.generate_summary(doc_type, extracted_fields, raw_text)

        # 9. Structured Record Construction
        doc_id = os.path.splitext(filename)[0]
        if not doc_id.startswith("DOC-"):
            doc_id = f"DOC-{abs(hash(filename)) % 100000:05d}"

        record = {
            "document_id": doc_id,
            "original_filename": filename,
            "file_path": file_path,
            "document_type": doc_type,
            "confidence": fusion_result["overall_confidence"],
            "fusion_details": fusion_result,
            "extracted_data": {
                "document_id": doc_id,
                "document_type": doc_type,
                "confidence": fusion_result["overall_confidence"],
                "fields": extracted_fields
            },
            "validation_results": val_result,
            "nlp_analysis": nlp_res,
            "summary": summary_text
        }

        # 10. Database Persistence
        self.db.insert_document(record)

        return record
