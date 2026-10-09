"""
Document Upload and Management Module (Phase 1)
Handles file validation, hashing, unique ID generation, and disk storage.
"""

import os
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}

class DocumentManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = os.path.abspath(base_dir)
        self.raw_dir = os.path.join(self.base_dir, "raw")
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.outputs_dir = os.path.join(self.base_dir, "outputs")
        
        for d in [self.raw_dir, self.processed_dir, self.outputs_dir]:
            os.makedirs(d, exist_ok=True)

    def validate_file(self, filename: str, file_bytes: bytes) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{ext}'. Supported: {', '.join(ALLOWED_EXTENSIONS)}")
        if len(file_bytes) == 0:
            raise ValueError("File is empty (0 bytes).")
        return True

    def save_upload(self, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        self.validate_file(filename, file_bytes)
        
        doc_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        ext = os.path.splitext(filename)[1].lower()
        saved_filename = f"{doc_id}_{filename}"
        file_path = os.path.join(self.raw_dir, saved_filename)
        
        with open(file_path, "wb") as f:
            f.write(file_bytes)
            
        return {
            "document_id": doc_id,
            "original_filename": filename,
            "file_path": file_path,
            "file_extension": ext,
            "file_size": len(file_bytes),
            "sha256": file_hash,
            "uploaded_at": datetime.now().isoformat(),
            "status": "UPLOADED"
        }
