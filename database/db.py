"""
Database Models & Persistence Layer (Phase 22)
Handles document record storage, extraction JSON, classification metadata, and history retrieval.
Supports MongoDB with automatic local JSON/SQLite fallback.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class DocumentRecord(BaseModel):
    document_id: str
    original_filename: str
    file_path: str
    document_type: str
    confidence: float
    extracted_data: Dict[str, Any]
    validation_results: Dict[str, Any]
    summary: Optional[str] = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class DatabaseManager:
    def __init__(self, db_path: str = "data/database.db", mongo_uri: str = "mongodb://localhost:27017"):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.use_mongo = False
        
        try:
            from pymongo import MongoClient
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=500)
            self.mongo_client.server_info()
            self.db = self.mongo_client["document_intelligence_db"]
            self.collection = self.db["documents"]
            self.use_mongo = True
        except Exception:
            self.use_mongo = False
            self._init_sqlite()

    def _init_sqlite(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                original_filename TEXT,
                file_path TEXT,
                document_type TEXT,
                confidence REAL,
                extracted_data TEXT,
                validation_results TEXT,
                summary TEXT,
                created_at TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def insert_document(self, record: Dict[str, Any]) -> str:
        doc_id = record["document_id"]
        
        if self.use_mongo:
            self.collection.update_one(
                {"document_id": doc_id},
                {"$set": record},
                upsert=True
            )
            return doc_id
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (document_id, original_filename, file_path, document_type, confidence, extracted_data, validation_results, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_id,
                record.get("original_filename", ""),
                record.get("file_path", ""),
                record.get("document_type", "unknown"),
                record.get("confidence", 0.0),
                json.dumps(record.get("extracted_data", {})),
                json.dumps(record.get("validation_results", {})),
                record.get("summary", ""),
                record.get("created_at", datetime.now().isoformat())
            ))
            conn.commit()
            conn.close()
            return doc_id

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        if self.use_mongo:
            doc = self.collection.find_one({"document_id": document_id})
            if doc and "_id" in doc:
                del doc["_id"]
            return doc
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM documents WHERE document_id = ?', (document_id,))
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            return {
                "document_id": row[0],
                "original_filename": row[1],
                "file_path": row[2],
                "document_type": row[3],
                "confidence": row[4],
                "extracted_data": json.loads(row[5]),
                "validation_results": json.loads(row[6]),
                "summary": row[7],
                "created_at": row[8]
            }

    def list_documents(self, limit: int = 20) -> List[Dict[str, Any]]:
        if self.use_mongo:
            cursor = self.collection.find().sort("created_at", -1).limit(limit)
            docs = []
            for doc in cursor:
                if "_id" in doc:
                    del doc["_id"]
                docs.append(doc)
            return docs
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM documents ORDER BY created_at DESC LIMIT ?', (limit,))
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    "document_id": r[0],
                    "original_filename": r[1],
                    "file_path": r[2],
                    "document_type": r[3],
                    "confidence": r[4],
                    "extracted_data": json.loads(r[5]),
                    "validation_results": json.loads(r[6]),
                    "summary": r[7],
                    "created_at": r[8]
                }
                for r in rows
            ]
