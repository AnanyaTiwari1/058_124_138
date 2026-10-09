"""
PDF and Image Preprocessing Module (Phase 2)
Handles PDF-to-image extraction and image enhancements (Otsu thresholding, de-noising, contrast adjustment).
"""

import os
from typing import List, Tuple
from PIL import Image
import numpy as np
import cv2

class ImageEnhancer:
    @staticmethod
    def enhance_image(image_path: str, output_path: str = None) -> Tuple[str, np.ndarray]:
        """
        Enhances document image:
        1. Grayscale conversion
        2. Bilateral filtering for noise removal while preserving edges
        3. Otsu thresholding / adaptive contrast adjustment
        """
        if image_path.lower().endswith('.pdf'):
            # Convert first page to image for vision processing
            temp_dir = os.path.join(os.path.dirname(image_path), "pdf_renders")
            pages = PDFProcessor.convert_pdf_to_images(image_path, temp_dir, max_pages=1)
            if pages and os.path.exists(pages[0]):
                image_path = pages[0]

        img = cv2.imread(image_path)
        if img is None:
            try:
                pil_img = Image.open(image_path).convert('RGB')
                img = np.array(pil_img)[:, :, ::-1]
            except Exception:
                # Return dummy blank array if unreadable
                blank = np.ones((200, 200), dtype=np.uint8) * 255
                return image_path, blank

        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive contrast & thresholding
        enhanced = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        )
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, enhanced)
            return output_path, enhanced
            
        return image_path, enhanced

class PDFProcessor:
    @staticmethod
    def is_pdf(file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')

    @staticmethod
    def get_pdf_page_count(pdf_path: str) -> int:
        """Returns total page count of a PDF."""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except Exception:
            pass
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            return len(reader.pages)
        except Exception:
            return 1

    @staticmethod
    def convert_pdf_to_images(pdf_path: str, output_dir: str, max_pages: int = 20) -> List[str]:
        """
        Converts PDF pages into high-resolution PNG images for downstream OCR/Vision models.
        """
        os.makedirs(output_dir, exist_ok=True)
        images = []
        
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            total = min(len(doc), max_pages)
            for page_num in range(total):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=200)
                img_path = os.path.join(output_dir, f"page_{page_num+1}.png")
                pix.save(img_path)
                images.append(img_path)
            doc.close()
            return images
        except Exception:
            pass

        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=max_pages)
            for idx, page in enumerate(pages):
                img_path = os.path.join(output_dir, f"page_{idx+1}.png")
                page.save(img_path, "PNG")
                images.append(img_path)
            return images
        except Exception:
            return []

    @staticmethod
    def extract_text_from_all_pages(pdf_path: str) -> str:
        """Extracts text streams from all pages in a multi-page PDF."""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            full_text = []
            for idx, page in enumerate(doc):
                txt = page.get_text()
                if txt.strip():
                    full_text.append(f"--- Page {idx+1} ---\n{txt}")
            doc.close()
            if full_text:
                return "\n\n".join(full_text)
        except Exception:
            pass

        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            full_text = []
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    full_text.append(f"--- Page {idx+1} ---\n{txt}")
            if full_text:
                return "\n\n".join(full_text)
        except Exception:
            pass

        return ""
