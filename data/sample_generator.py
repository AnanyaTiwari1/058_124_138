"""
Sample Document Generator for Multimodal Document Intelligence Platform
Generates realistic sample images (Invoices, Receipts, Resumes, ID Cards) with clear text for testing OCR and multimodal processing.
"""

import os
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(DATA_DIR, "sample_documents")

def get_font(size=16, bold=False):
    try:
        # Standard Windows fonts
        font_path = "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf"
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
    except Exception:
        pass
    return ImageFont.load_default()

def create_invoice_sample(path: str):
    img = Image.new('RGB', (800, 1000), color='#ffffff')
    draw = ImageDraw.Draw(img)
    
    font_title = get_font(28, bold=True)
    font_header = get_font(18, bold=True)
    font_body = get_font(14, bold=False)
    font_bold = get_font(14, bold=True)
    
    # Header
    draw.rectangle([(0, 0), (800, 100)], fill='#1e293b')
    draw.text((40, 32), "TECHFLOW SOLUTIONS PVT LTD", fill='#ffffff', font=font_title)
    draw.text((620, 40), "TAX INVOICE", fill='#38bdf8', font=font_header)
    
    # Metadata
    y = 130
    draw.text((40, y), "Invoice Number: INV-2026-1023", fill='#0f172a', font=font_bold)
    draw.text((500, y), "Date: 15 August 2026", fill='#0f172a', font=font_bold)
    
    y += 30
    draw.text((40, y), "Seller: TechFlow Solutions Pvt Ltd", fill='#334155', font=font_body)
    draw.text((500, y), "Payment Terms: Due on Receipt", fill='#334155', font=font_body)
    
    y += 25
    draw.text((40, y), "GSTIN: 24ABCDE1234F1Z5", fill='#334155', font=font_body)
    draw.text((500, y), "Place of Supply: Gujarat (24)", fill='#334155', font=font_body)
    
    y += 40
    # Customer Details Box
    draw.rectangle([(40, y), (760, y + 80)], outline='#cbd5e1', width=1, fill='#f8fafc')
    draw.text((55, y + 10), "Billed To (Customer):", fill='#64748b', font=get_font(12, bold=True))
    draw.text((55, y + 30), "Rahul Shah", fill='#0f172a', font=font_header)
    draw.text((55, y + 55), "Email: rahul.shah@example.com | Phone: +91-9876543210", fill='#475569', font=font_body)
    
    y += 110
    # Table Header
    draw.rectangle([(40, y), (760, y + 35)], fill='#e2e8f0')
    draw.text((55, y + 8), "#", fill='#0f172a', font=font_bold)
    draw.text((90, y + 8), "Item Description", fill='#0f172a', font=font_bold)
    draw.text((440, y + 8), "Qty", fill='#0f172a', font=font_bold)
    draw.text((520, y + 8), "Unit Price", fill='#0f172a', font=font_bold)
    draw.text((660, y + 8), "Amount (INR)", fill='#0f172a', font=font_bold)
    
    # Items
    items = [
        ("1", "Enterprise AI Server License", "1", "50000.00", "50000.00"),
        ("2", "Cloud Storage & Backup (1TB)", "2", "7500.00", "15000.00"),
        ("3", "Premium Technical Support (1 Yr)", "1", "11500.00", "11500.00")
    ]
    
    y += 35
    for item in items:
        draw.line([(40, y + 35), (760, y + 35)], fill='#e2e8f0', width=1)
        draw.text((55, y + 8), item[0], fill='#334155', font=font_body)
        draw.text((90, y + 8), item[1], fill='#0f172a', font=font_body)
        draw.text((445, y + 8), item[2], fill='#334155', font=font_body)
        draw.text((520, y + 8), f"INR {item[3]}", fill='#334155', font=font_body)
        draw.text((660, y + 8), f"INR {item[4]}", fill='#0f172a', font=font_bold)
        y += 38
        
    y += 30
    # Totals Box
    draw.line([(450, y), (760, y)], fill='#94a3b8', width=1)
    y += 10
    draw.text((450, y), "Subtotal:", fill='#475569', font=font_body)
    draw.text((660, y), "INR 76500.00", fill='#0f172a', font=font_bold)
    
    y += 25
    draw.text((450, y), "Tax / GST (18%):", fill='#475569', font=font_body)
    draw.text((660, y), "INR 13770.00", fill='#0f172a', font=font_bold)
    
    y += 30
    draw.rectangle([(440, y), (760, y + 40)], fill='#1e293b')
    draw.text((450, y + 10), "Grand Total:", fill='#ffffff', font=font_header)
    draw.text((640, y + 10), "INR 90270.00", fill='#38bdf8', font=font_header)
    
    # Footer note
    y = 920
    draw.text((40, y), "Thank you for your business! For queries contact billing@techflow.com", fill='#94a3b8', font=get_font(12))
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print(f"Generated invoice sample at {path}")

def create_receipt_sample(path: str):
    img = Image.new('RGB', (500, 750), color='#fffdfa')
    draw = ImageDraw.Draw(img)
    
    font_title = get_font(20, bold=True)
    font_bold = get_font(14, bold=True)
    font_body = get_font(13, bold=False)
    
    y = 30
    draw.text((140, y), "COFFEE & BISTRO", fill='#1e293b', font=font_title)
    y += 30
    draw.text((155, y), "MG Road, Bengaluru", fill='#64748b', font=font_body)
    y += 20
    draw.text((170, y), "GST: 29AABCB1234F1Z1", fill='#64748b', font=get_font(11))
    
    y += 35
    draw.line([(30, y), (470, y)], fill='#cbd5e1', width=1)
    y += 15
    draw.text((30, y), "Date: 2026-08-20 14:32", fill='#334155', font=font_body)
    draw.text((320, y), "Order #482", fill='#334155', font=font_bold)
    
    y += 25
    draw.text((30, y), "Merchant: Aroma Cafe", fill='#334155', font=font_body)
    
    y += 30
    draw.line([(30, y), (470, y)], fill='#cbd5e1', width=1)
    y += 15
    
    items = [
        ("Cappuccino (Large)", "2", "380.00"),
        ("Blueberry Muffin", "1", "180.00"),
        ("Veg Club Sandwich", "1", "240.00"),
        ("Mineral Water", "2", "60.00")
    ]
    
    for item, qty, amt in items:
        draw.text((30, y), f"{qty}x {item}", fill='#0f172a', font=font_body)
        draw.text((400, y), f"INR {amt}", fill='#0f172a', font=font_body)
        y += 26
        
    y += 20
    draw.line([(30, y), (470, y)], fill='#cbd5e1', width=1)
    y += 15
    
    draw.text((250, y), "Subtotal:", fill='#475569', font=font_body)
    draw.text((400, y), "INR 860.00", fill='#0f172a', font=font_body)
    y += 22
    draw.text((250, y), "CGST + SGST (5%):", fill='#475569', font=font_body)
    draw.text((400, y), "INR 43.00", fill='#0f172a', font=font_body)
    y += 28
    draw.text((250, y), "Total Amount:", fill='#0f172a', font=font_bold)
    draw.text((390, y), "INR 903.00", fill='#059669', font=font_bold)
    
    y += 45
    draw.line([(30, y), (470, y)], fill='#cbd5e1', width=1)
    y += 25
    draw.text((150, y), "Payment: UPI / Card Approved", fill='#64748b', font=font_body)
    y += 25
    draw.text((170, y), "*** THANK YOU ***", fill='#1e293b', font=font_bold)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print(f"Generated receipt sample at {path}")

def create_resume_sample(path: str):
    img = Image.new('RGB', (800, 1050), color='#ffffff')
    draw = ImageDraw.Draw(img)
    
    font_name = get_font(26, bold=True)
    font_role = get_font(16, bold=True)
    font_section = get_font(16, bold=True)
    font_body = get_font(13, bold=False)
    font_sub = get_font(13, bold=True)
    
    # Header Banner
    draw.rectangle([(0, 0), (800, 120)], fill='#0f172a')
    draw.text((40, 25), "ANANYA VERMA", fill='#ffffff', font=font_name)
    draw.text((40, 65), "Senior AI & Machine Learning Engineer", fill='#38bdf8', font=font_role)
    draw.text((40, 92), "Email: ananya.verma@email.com | Phone: +91-9811223344 | Bengaluru, India", fill='#94a3b8', font=get_font(12))
    
    y = 150
    # Summary
    draw.text((40, y), "PROFESSIONAL SUMMARY", fill='#0284c7', font=font_section)
    draw.line([(40, y + 24), (760, y + 24)], fill='#0284c7', width=2)
    y += 35
    summary = "Accomplished Machine Learning Engineer with 5+ years of experience designing deep learning models, multimodal document AI pipelines, NLP architectures (BERT, GPT), and deploying production REST APIs."
    draw.text((40, y), summary, fill='#334155', font=font_body)
    
    y += 60
    # Skills
    draw.text((40, y), "TECHNICAL SKILLS", fill='#0284c7', font=font_section)
    draw.line([(40, y + 24), (760, y + 24)], fill='#0284c7', width=2)
    y += 35
    draw.text((40, y), "• Deep Learning: PyTorch, TensorFlow, CNN, RNN, LSTM, Autoencoders, GAN, ResNet", fill='#334155', font=font_body)
    y += 24
    draw.text((40, y), "• NLP & LLMs: SpaCy, NLTK, Transformers, Word2Vec, GloVe, TF-IDF, NER, POS Tagging", fill='#334155', font=font_body)
    y += 24
    draw.text((40, y), "• Backend & Cloud: Python, FastAPI, Flask, MongoDB, Docker, AWS, Streamlit", fill='#334155', font=font_body)
    
    y += 50
    # Experience
    draw.text((40, y), "WORK EXPERIENCE", fill='#0284c7', font=font_section)
    draw.line([(40, y + 24), (760, y + 24)], fill='#0284c7', width=2)
    y += 35
    draw.text((40, y), "Lead AI Engineer | CogniTech Solutions (2023 - Present)", fill='#0f172a', font=font_sub)
    y += 22
    draw.text((40, y), "• Architected multimodal document extraction engines handling over 50,000 invoices/day.", fill='#475569', font=font_body)
    y += 20
    draw.text((40, y), "• Deployed transformer-based question answering chatbots with 94% precision.", fill='#475569', font=font_body)
    
    y += 35
    draw.text((40, y), "Machine Learning Engineer | DataCore Labs (2021 - 2023)", fill='#0f172a', font=font_sub)
    y += 22
    draw.text((40, y), "• Developed CNN and ResNet models for document classification achieving 97.2% accuracy.", fill='#475569', font=font_body)
    y += 20
    draw.text((40, y), "• Implemented custom NER pipelines and TF-IDF term extractors for resume parsing.", fill='#475569', font=font_body)
    
    y += 50
    # Education
    draw.text((40, y), "EDUCATION", fill='#0284c7', font=font_section)
    draw.line([(40, y + 24), (760, y + 24)], fill='#0284c7', width=2)
    y += 35
    draw.text((40, y), "Master of Technology (M.Tech) in Computer Science & AI (2019 - 2021)", fill='#0f172a', font=font_sub)
    y += 20
    draw.text((40, y), "Indian Institute of Technology (IIT) | CGPA: 9.2/10", fill='#475569', font=font_body)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print(f"Generated resume sample at {path}")

def create_id_card_sample(path: str):
    img = Image.new('RGB', (650, 400), color='#f1f5f9')
    draw = ImageDraw.Draw(img)
    
    # Outer frame
    draw.rectangle([(20, 20), (630, 380)], fill='#ffffff', outline='#94a3b8', width=2)
    # Header bar
    draw.rectangle([(20, 20), (630, 85)], fill='#1e3a8a')
    draw.text((40, 35), "GOVERNMENT IDENTITY CARD", fill='#ffffff', font=get_font(20, bold=True))
    draw.text((40, 62), "Republic of India / National Identity Authority", fill='#93c5fd', font=get_font(11))
    
    # Photo placeholder
    draw.rectangle([(50, 115), (190, 290)], fill='#e2e8f0', outline='#64748b', width=1)
    draw.text((95, 190), "[ PHOTO ]", fill='#475569', font=get_font(12, bold=True))
    
    # Details
    x = 220
    y = 115
    draw.text((x, y), "Full Name:", fill='#64748b', font=get_font(12))
    draw.text((x, y + 16), "VIKRAM SINGH", fill='#0f172a', font=get_font(16, bold=True))
    
    y += 45
    draw.text((x, y), "Date of Birth (DOB):", fill='#64748b', font=get_font(12))
    draw.text((x, y + 16), "12/06/1995", fill='#0f172a', font=get_font(14, bold=True))
    
    draw.text((x + 180, y), "Gender:", fill='#64748b', font=get_font(12))
    draw.text((x + 180, y + 16), "Male", fill='#0f172a', font=get_font(14, bold=True))
    
    y += 45
    draw.text((x, y), "Identity Number:", fill='#64748b', font=get_font(12))
    draw.text((x, y + 16), "ABCD-9876-5432", fill='#b91c1c', font=get_font(17, bold=True))
    
    y += 45
    draw.text((x, y), "Address:", fill='#64748b', font=get_font(12))
    draw.text((x, y + 16), "42, Shanti Nagar, MG Road, New Delhi - 110001", fill='#334155', font=get_font(12))
    
    # Bottom bar
    draw.rectangle([(20, 340), (630, 380)], fill='#e2e8f0')
    draw.text((200, 352), "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", fill='#334155', font=get_font(14))
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print(f"Generated ID card sample at {path}")

def generate_all_samples():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    create_invoice_sample(os.path.join(SAMPLE_DIR, "invoice_sample.jpg"))
    create_receipt_sample(os.path.join(SAMPLE_DIR, "receipt_sample.jpg"))
    create_resume_sample(os.path.join(SAMPLE_DIR, "resume_sample.jpg"))
    create_id_card_sample(os.path.join(SAMPLE_DIR, "id_card_sample.jpg"))

if __name__ == "__main__":
    generate_all_samples()
