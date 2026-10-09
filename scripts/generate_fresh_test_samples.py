"""
Generate Fresh Test Documents (2 PDFs and 1 JPG)
Outputs:
1. data/test_samples/commercial_tax_invoice.pdf (2-Page Professional Tax Invoice PDF)
2. data/test_samples/ai_research_scientist_resume.pdf (2-Page Professional Resume PDF)
3. data/test_samples/supermarket_retail_receipt.jpg (High-res Scanned Retail Receipt JPG)
"""

import os
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "test_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Commercial Tax Invoice PDF (2 Pages)
# -------------------------------------------------------------
def create_invoice_pdf():
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page(width=595, height=842) # A4 size
    rect = fitz.Rect(40, 40, 555, 802)
    
    # Header
    page1.draw_rect(fitz.Rect(40, 40, 555, 110), color=(0.1, 0.2, 0.45), fill=(0.93, 0.95, 0.99))
    page1.insert_text((55, 70), "NEXUS CLOUD & AI SOLUTIONS PVT LTD", fontsize=15, color=(0.1, 0.2, 0.5), fontname="helv")
    page1.insert_text((55, 90), "TAX INVOICE | GSTIN: 27AABCN8899F1Z4 | CIN: U72200MH2021PTC123456", fontsize=9, color=(0.3, 0.3, 0.3))
    page1.insert_text((55, 102), "Address: Level 8, Cyber Tower, Bandra Kurla Complex, Mumbai - 400051", fontsize=8, color=(0.4, 0.4, 0.4))
    
    # Invoice Metadata Box
    page1.draw_rect(fitz.Rect(40, 125, 290, 200), color=(0.7, 0.7, 0.7))
    page1.insert_text((50, 145), "INVOICE DETAILS", fontsize=10, color=(0.1, 0.2, 0.5))
    page1.insert_text((50, 162), "Invoice Number: INV-2026-9844", fontsize=9, color=(0, 0, 0))
    page1.insert_text((50, 177), "Invoice Date: 24 September 2026", fontsize=9, color=(0, 0, 0))
    page1.insert_text((50, 192), "Due Date: 10 October 2026 (Net 15)", fontsize=9, color=(0, 0, 0))
    
    # Customer Billed To Box
    page1.draw_rect(fitz.Rect(305, 125, 555, 200), color=(0.7, 0.7, 0.7))
    page1.insert_text((315, 145), "BILLED TO (CUSTOMER)", fontsize=10, color=(0.1, 0.2, 0.5))
    page1.insert_text((315, 162), "Customer Name: Rohan Malhotra", fontsize=9, color=(0, 0, 0))
    page1.insert_text((315, 177), "Company: Apex FinTech Technologies", fontsize=9, color=(0, 0, 0))
    page1.insert_text((315, 192), "Email: rohan.malhotra@apexfin.com | +91-9820011223", fontsize=8, color=(0, 0, 0))
    
    # Line Items Table Header
    page1.draw_rect(fitz.Rect(40, 220, 555, 245), color=(0.1, 0.2, 0.5), fill=(0.1, 0.2, 0.5))
    page1.insert_text((50, 237), "Item Description", fontsize=9, color=(1, 1, 1))
    page1.insert_text((280, 237), "Qty", fontsize=9, color=(1, 1, 1))
    page1.insert_text((340, 237), "Unit Price (INR)", fontsize=9, color=(1, 1, 1))
    page1.insert_text((460, 237), "Amount (INR)", fontsize=9, color=(1, 1, 1))
    
    # Table Rows
    rows = [
        ("1. Dedicated GPU Cluster (NVIDIA H100 80GB x4)", "1", "120000.00", "120000.00"),
        ("2. Enterprise LLM Fine-Tuning Infrastructure", "1", "45000.00", "45000.00"),
        ("3. Multimodal OCR & Document API Gateway", "2", "12500.00", "25000.00"),
        ("4. Platinum 24x7 SLA & Incident Management Support", "1", "20000.00", "20000.00")
    ]
    y = 265
    for desc, qty, up, amt in rows:
        page1.insert_text((50, y), desc, fontsize=8.5, color=(0.1, 0.1, 0.1))
        page1.insert_text((290, y), qty, fontsize=8.5, color=(0.1, 0.1, 0.1))
        page1.insert_text((360, y), up, fontsize=8.5, color=(0.1, 0.1, 0.1))
        page1.insert_text((475, y), amt, fontsize=8.5, color=(0.1, 0.1, 0.1))
        page1.draw_line(fitz.Point(40, y + 8), fitz.Point(555, y + 8), color=(0.85, 0.85, 0.85))
        y += 28
        
    # Totals Box
    page1.draw_rect(fitz.Rect(320, 390, 555, 480), color=(0.7, 0.7, 0.7), fill=(0.97, 0.98, 1.0))
    page1.insert_text((330, 412), "Subtotal:", fontsize=9.5, color=(0.2, 0.2, 0.2))
    page1.insert_text((460, 412), "INR 210000.00", fontsize=9.5, color=(0.1, 0.1, 0.1))
    
    page1.insert_text((330, 432), "CGST (9%) + SGST (9%):", fontsize=9.5, color=(0.2, 0.2, 0.2))
    page1.insert_text((460, 432), "INR 37800.00", fontsize=9.5, color=(0.1, 0.1, 0.1))
    
    page1.draw_line(fitz.Point(325, 445), fitz.Point(550, 445), color=(0.6, 0.6, 0.6))
    page1.insert_text((330, 465), "Grand Total:", fontsize=11, color=(0.1, 0.2, 0.5), fontname="helv")
    page1.insert_text((455, 465), "INR 247800.00", fontsize=11, color=(0.1, 0.2, 0.5), fontname="helv")
    
    page1.insert_text((50, 520), "Amount in words: Two Lakh Forty-Seven Thousand Eight Hundred Rupees Only", fontsize=8.5, color=(0.3, 0.3, 0.3))
    page1.insert_text((50, 780), "Page 1 of 2 — Continued on next page...", fontsize=8, color=(0.5, 0.5, 0.5))

    # Page 2 (Terms & Wire Transfer)
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 60), "NEXUS CLOUD & AI SOLUTIONS — TERMS & WIRE DETAILS", fontsize=12, color=(0.1, 0.2, 0.5), fontname="helv")
    page2.draw_line(fitz.Point(50, 70), fitz.Point(545, 70), color=(0.1, 0.2, 0.5))
    
    page2.insert_text((50, 100), "Bank Account Details for Wire Transfer / NEFT:", fontsize=10, color=(0.1, 0.1, 0.1), fontname="helv")
    page2.insert_text((50, 120), "• Account Name: Nexus Cloud Solutions Private Limited", fontsize=9, color=(0.2, 0.2, 0.2))
    page2.insert_text((50, 138), "• Bank Name: HDFC Bank Ltd (Bandra Kurla Branch)", fontsize=9, color=(0.2, 0.2, 0.2))
    page2.insert_text((50, 156), "• Account Number: 50200098765432", fontsize=9, color=(0.2, 0.2, 0.2))
    page2.insert_text((50, 174), "• IFSC Code: HDFC0001234 | Swift Code: HDFCINBBXXX", fontsize=9, color=(0.2, 0.2, 0.2))
    
    page2.insert_text((50, 210), "Standard Commercial Terms & Conditions:", fontsize=10, color=(0.1, 0.1, 0.1), fontname="helv")
    page2.insert_text((50, 230), "1. Payments received after the due date will incur a late interest charge of 1.5% per month.", fontsize=8.5, color=(0.3, 0.3, 0.3))
    page2.insert_text((50, 248), "2. All services are governed by the Master Service Agreement (MSA-2026) executed between parties.", fontsize=8.5, color=(0.3, 0.3, 0.3))
    page2.insert_text((50, 266), "3. Disputes are subject to the exclusive jurisdiction of courts in Mumbai, Maharashtra.", fontsize=8.5, color=(0.3, 0.3, 0.3))
    
    # Authorized Signatory
    page2.draw_rect(fitz.Rect(350, 320, 530, 400), color=(0.6, 0.6, 0.6))
    page2.insert_text((370, 345), "Authorized Signatory", fontsize=9, color=(0.2, 0.2, 0.2))
    page2.insert_text((370, 375), "[DIGITALLY SIGNED & VERIFIED]", fontsize=8, color=(0.1, 0.5, 0.2))
    page2.insert_text((370, 390), "Nexus Cloud Solutions Pvt Ltd", fontsize=7.5, color=(0.4, 0.4, 0.4))
    
    page2.insert_text((50, 780), "Page 2 of 2 — End of Invoice Document", fontsize=8, color=(0.5, 0.5, 0.5))

    inv_path = os.path.join(OUTPUT_DIR, "commercial_tax_invoice.pdf")
    doc.save(inv_path)
    doc.close()
    print(f"[+] Generated Invoice PDF: {inv_path}")
    return inv_path

# -------------------------------------------------------------
# 2. AI Research Scientist Resume PDF (2 Pages)
# -------------------------------------------------------------
def create_resume_pdf():
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page(width=595, height=842)
    
    # Header
    page1.draw_rect(fitz.Rect(40, 40, 555, 115), color=(0.15, 0.25, 0.35), fill=(0.95, 0.97, 0.98))
    page1.insert_text((55, 72), "DR. SIDDHARTH MEHTA", fontsize=18, color=(0.1, 0.2, 0.35), fontname="helv")
    page1.insert_text((55, 92), "Principal AI & NLP Research Scientist | Deep Learning Architect", fontsize=10, color=(0.2, 0.4, 0.6))
    page1.insert_text((55, 107), "Email: siddharth.mehta@ai-research.org | Phone: +91-9871122334 | Bengaluru, India", fontsize=8.5, color=(0.3, 0.3, 0.3))
    
    # Professional Summary
    page1.insert_text((45, 140), "EXECUTIVE SUMMARY", fontsize=11, color=(0.1, 0.2, 0.35), fontname="helv")
    page1.draw_line(fitz.Point(45, 145), fitz.Point(550, 145), color=(0.2, 0.4, 0.6))
    page1.insert_textbox(fitz.Rect(45, 152, 550, 205),
        "Pioneering Deep Learning Researcher with 7+ years of expertise leading multimodal foundation models, "
        "document intelligence systems, and transformer-based LLM architectures. Author of 12 peer-reviewed papers "
        "in NeurIPS, CVPR, and ACL. Expert in PyTorch, Distributed Training, and End-to-End Multimodal Vision-Language Systems.",
        fontsize=9, color=(0.2, 0.2, 0.2)
    )
    
    # Technical Skills
    page1.insert_text((45, 220), "TECHNICAL SKILLS & COMPETENCIES", fontsize=11, color=(0.1, 0.2, 0.35), fontname="helv")
    page1.draw_line(fitz.Point(45, 225), fitz.Point(550, 225), color=(0.2, 0.4, 0.6))
    page1.insert_text((50, 242), "• Deep Learning: PyTorch, TensorFlow, Torchvision, CNN, Bi-LSTM, Autoencoders, VAE, GANs", fontsize=9, color=(0.15, 0.15, 0.15))
    page1.insert_text((50, 258), "• NLP & Transformers: BERT, RoBERTa, GPT, Word2Vec, GloVe, TF-IDF, POS Tagging, NER, N-Grams", fontsize=9, color=(0.15, 0.15, 0.15))
    page1.insert_text((50, 274), "• Computer Vision: YOLOv8, ResNet50, VGG16, LayoutLMv3, OpenCV, PyMuPDF, OCR Engines", fontsize=9, color=(0.15, 0.15, 0.15))
    page1.insert_text((50, 290), "• Cloud & APIs: FastAPI, Docker, Kubernetes, MongoDB, Redis, Streamlit, Groq LLM Cloud API", fontsize=9, color=(0.15, 0.15, 0.15))
    
    # Work Experience
    page1.insert_text((45, 320), "PROFESSIONAL EXPERIENCE", fontsize=11, color=(0.1, 0.2, 0.35), fontname="helv")
    page1.draw_line(fitz.Point(45, 325), fitz.Point(550, 325), color=(0.2, 0.4, 0.6))
    
    page1.insert_text((50, 345), "Principal AI Scientist | Cortex Intelligence Labs (2023 - Present)", fontsize=10, color=(0.1, 0.2, 0.35), fontname="helv")
    page1.insert_text((50, 362), "• Spearheaded the development of multimodal document classification systems processing 2M+ docs/month.", fontsize=8.5, color=(0.25, 0.25, 0.25))
    page1.insert_text((50, 378), "• Reduced OCR error propagation by 38% using custom Convolutional Autoencoder (CAE) denoising filters.", fontsize=8.5, color=(0.25, 0.25, 0.25))
    
    page1.insert_text((50, 410), "Senior Machine Learning Engineer | DeepVision AI (2020 - 2023)", fontsize=10, color=(0.1, 0.2, 0.35), fontname="helv")
    page1.insert_text((50, 427), "• Designed Bi-LSTM sequence labeling pipelines achieving 96.4% F1-score for tabular invoice parsing.", fontsize=8.5, color=(0.25, 0.25, 0.25))
    page1.insert_text((50, 443), "• Implemented multimodal late-fusion architectures combining visual ResNet embeddings and BERT text features.", fontsize=8.5, color=(0.25, 0.25, 0.25))
    
    page1.insert_text((50, 780), "Page 1 of 2 — Continued on next page...", fontsize=8, color=(0.5, 0.5, 0.5))

    # Page 2 (Education & Publications)
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((45, 60), "DR. SIDDHARTH MEHTA — EDUCATION & PUBLICATIONS", fontsize=12, color=(0.1, 0.2, 0.35), fontname="helv")
    page2.draw_line(fitz.Point(45, 68), fitz.Point(550, 68), color=(0.2, 0.4, 0.6))
    
    # Education
    page2.insert_text((45, 95), "EDUCATION & ACADEMIC CREDENTIALS", fontsize=11, color=(0.1, 0.2, 0.35), fontname="helv")
    page2.draw_line(fitz.Point(45, 100), fitz.Point(550, 100), color=(0.2, 0.4, 0.6))
    
    page2.insert_text((50, 120), "Ph.D. in Computer Science & Artificial Intelligence (2016 - 2020)", fontsize=9.5, color=(0.1, 0.1, 0.1), fontname="helv")
    page2.insert_text((50, 135), "Indian Institute of Science (IISc), Bangalore | Focus: Multimodal Representation Learning", fontsize=8.5, color=(0.3, 0.3, 0.3))
    
    page2.insert_text((50, 160), "B.Tech in Computer Engineering (2012 - 2016)", fontsize=9.5, color=(0.1, 0.1, 0.1), fontname="helv")
    page2.insert_text((50, 175), "National Institute of Technology (NIT) | First Class with Distinction (GPA: 9.4/10)", fontsize=8.5, color=(0.3, 0.3, 0.3))
    
    # Selected Publications
    page2.insert_text((45, 210), "KEY RESEARCH PUBLICATIONS", fontsize=11, color=(0.1, 0.2, 0.35), fontname="helv")
    page2.draw_line(fitz.Point(45, 215), fitz.Point(550, 215), color=(0.2, 0.4, 0.6))
    page2.insert_text((50, 235), "1. Mehta, S., et al. (2024). 'Confidence-Weighted Late Fusion in Document Understanding.' NeurIPS.", fontsize=8.5, color=(0.2, 0.2, 0.2))
    page2.insert_text((50, 255), "2. Mehta, S., et al. (2022). 'Variational Autoencoders for Scanned Document Inpainting.' CVPR.", fontsize=8.5, color=(0.2, 0.2, 0.2))
    page2.insert_text((50, 275), "3. Mehta, S., et al. (2021). 'Bi-LSTM Entity Extraction over Dense Tabular Layouts.' EMNLP.", fontsize=8.5, color=(0.2, 0.2, 0.2))
    
    page2.insert_text((50, 780), "Page 2 of 2 — End of Curriculum Vitae", fontsize=8, color=(0.5, 0.5, 0.5))

    res_path = os.path.join(OUTPUT_DIR, "ai_research_scientist_resume.pdf")
    doc.save(res_path)
    doc.close()
    print(f"[+] Generated Resume PDF: {res_path}")
    return res_path

# -------------------------------------------------------------
# 3. Supermarket Retail Receipt Image (JPG)
# -------------------------------------------------------------
def create_receipt_jpg():
    w, h = 600, 950
    img = Image.new('RGB', (w, h), color=(248, 248, 246))
    draw = ImageDraw.Draw(img)
    
    # Outer receipt border with paper texture feel
    draw.rectangle([20, 20, w - 20, h - 20], outline=(200, 200, 195), width=2)
    
    # Store Header
    draw.text((160, 45), "HYPERMARKET & DELI", fill=(20, 20, 20))
    draw.text((175, 75), "Phoenix MarketCity Mall, Pune", fill=(70, 70, 70))
    draw.text((190, 95), "GSTIN: 27AABCP4455G1Z9", fill=(80, 80, 80))
    draw.text((150, 115), "Date: 28-09-2026 19:42  Order #921", fill=(60, 60, 60))
    
    # Divider line
    draw.line([(40, 140), (w - 40, 140)], fill=(150, 150, 150), width=1)
    
    # Itemized Table
    draw.text((50, 155), "ITEM DESCRIPTION", fill=(30, 30, 30))
    draw.text((370, 155), "QTY", fill=(30, 30, 30))
    draw.text((470, 155), "PRICE (INR)", fill=(30, 30, 30))
    draw.line([(40, 175), (w - 40, 175)], fill=(180, 180, 180), width=1)
    
    items = [
        ("Organic Almond Milk 1L", "2", "360.00"),
        ("Belgian Dark Chocolate 100g", "3", "450.00"),
        ("Whole Wheat Sourdough Loaf", "1", "180.00"),
        ("Greek Yogurt Honey (400g)", "2", "240.00"),
        ("Arabica Roasted Coffee Beans", "1", "650.00"),
        ("Extra Virgin Olive Oil 500ml", "1", "720.00")
    ]
    
    y = 195
    for item, qty, price in items:
        draw.text((50, y), item, fill=(40, 40, 40))
        draw.text((380, y), qty, fill=(40, 40, 40))
        draw.text((485, y), price, fill=(40, 40, 40))
        y += 36
        
    draw.line([(40, y + 10), (w - 40, y + 10)], fill=(150, 150, 150), width=1)
    
    # Totals
    y += 25
    draw.text((300, y), "Subtotal:", fill=(50, 50, 50))
    draw.text((470, y), "INR 2600.00", fill=(30, 30, 30))
    
    y += 28
    draw.text((300, y), "CGST + SGST (5%):", fill=(50, 50, 50))
    draw.text((470, y), "INR 130.00", fill=(30, 30, 30))
    
    y += 32
    draw.line([(290, y), (w - 40, y)], fill=(80, 80, 80), width=2)
    y += 10
    draw.text((300, y), "TOTAL AMOUNT:", fill=(10, 10, 80))
    draw.text((460, y), "INR 2730.00", fill=(10, 10, 80))
    
    # Footer & Payment
    y += 55
    draw.text((50, y), "Payment Mode: UPI / PhonePe (TXN-9988221)", fill=(60, 60, 60))
    draw.text((50, y + 25), "Items Count: 6 | Total Units: 10", fill=(80, 80, 80))
    
    # Barcode representation
    y += 80
    for bx in range(120, 480, 6):
        draw.line([(bx, y), (bx, y + 50)], fill=(30, 30, 30), width=3 if bx % 12 == 0 else 1)
        
    draw.text((210, y + 60), "* 9 2 1 0 0 2 8 0 9 2 6 *", fill=(60, 60, 60))
    draw.text((180, y + 85), "*** THANK YOU FOR SHOPPING ***", fill=(50, 50, 50))

    receipt_path = os.path.join(OUTPUT_DIR, "supermarket_retail_receipt.jpg")
    img.save(receipt_path, "JPEG", quality=95)
    print(f"[+] Generated Receipt JPG: {receipt_path}")
    return receipt_path

if __name__ == "__main__":
    create_invoice_pdf()
    create_resume_pdf()
    create_receipt_jpg()
