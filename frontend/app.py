import os
import sys
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
from PIL import Image

# Import platform components
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
from experiments.nlpl.exp3_social_media_sentiment import run_social_media_sentiment_experiment
from experiments.nlpl.exp4_word2vec_glove_classifier import run_word_vector_classifier_experiment
from deep_learning.transfer_learning.transfer_model import TransferLearningEvaluator
from deep_learning.object_detection.yolo_detector import DocumentRegionDetector
from nlp.embeddings.word_embeddings import WordEmbeddingEngine

st.set_page_config(
    page_title="Multimodal Document Intelligence - GLS FET Sem VII",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic & Professional Aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f46e5, #06b6d4, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .course-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .course-nlpl {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .course-dll {
        background-color: #dbeafe;
        color: #1e40af;
        border: 1px solid #93c5fd;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_coordinator():
    return DocumentIntelligenceCoordinator()

coordinator = get_coordinator()
embed_engine = WordEmbeddingEngine()
transfer_evaluator = TransferLearningEvaluator()
region_detector = DocumentRegionDetector()

# Header
st.markdown('<div class="main-title">🧠 Multimodal Document Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown("""
<div>
    <span class="course-badge course-nlpl">📗 NLPL Course Code: 2604734</span>
    <span class="course-badge course-dll">📘 DLL Course Code: 2604733</span>
    <span style="color: #64748b; font-size: 0.9rem;">GLS FET B.Tech (CS&E) Semester VII — Track 3 (AI & ML)</span>
</div>
<hr style="margin-top: 10px; margin-bottom: 20px;"/>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "📄 Document Intelligence Processing", 
    "🟢 NLPL Laboratory (2604734)", 
    "🔵 Deep Learning Laboratory (2604733)", 
    "📚 Syllabus Mapping & 40-Session Plan"
])

# ==========================================
# TAB 1: Main Document Processing
# ==========================================
with tabs[0]:
    st.subheader("1. Document Ingestion & Multimodal Understanding")
    
    col_input, col_preview = st.columns([1, 1])
    
    sample_options = {
        "Sample Tax Invoice (INV-2026-1023)": "data/sample_documents/invoice_sample.jpg",
        "Sample Coffee Shop Receipt (Order #482)": "data/sample_documents/receipt_sample.jpg",
        "Sample AI Engineer Resume (Ananya Verma)": "data/sample_documents/resume_sample.jpg",
        "Sample National ID Card (Vikram Singh)": "data/sample_documents/id_card_sample.jpg"
    }
    
    with col_input:
        choice = st.radio("Choose Input Method:", ["Select Pre-loaded Academic Sample", "Upload Custom Document (PDF / Image)"])
        target_path = None
        
        if choice == "Select Pre-loaded Academic Sample":
            selected_sample = st.selectbox("Select Academic Test Sample:", list(sample_options.keys()))
            target_path = sample_options[selected_sample]
        else:
            with st.expander("📦 Download Test Files for Manual Upload", expanded=False):
                st.caption("Download these fresh test files to your computer, then drag & drop them below:")
                dl_c1, dl_c2, dl_c3 = st.columns(3)
                
                inv_sample_pdf = "data/test_samples/commercial_tax_invoice.pdf"
                if os.path.exists(inv_sample_pdf):
                    with open(inv_sample_pdf, "rb") as f:
                        dl_c1.download_button("📄 Invoice (PDF)", f.read(), file_name="commercial_tax_invoice.pdf", mime="application/pdf")
                        
                res_sample_pdf = "data/test_samples/ai_research_scientist_resume.pdf"
                if os.path.exists(res_sample_pdf):
                    with open(res_sample_pdf, "rb") as f:
                        dl_c2.download_button("📄 Resume (PDF)", f.read(), file_name="ai_research_scientist_resume.pdf", mime="application/pdf")
                        
                rec_sample_jpg = "data/test_samples/supermarket_retail_receipt.jpg"
                if os.path.exists(rec_sample_jpg):
                    with open(rec_sample_jpg, "rb") as f:
                        dl_c3.download_button("🖼️ Receipt (JPG)", f.read(), file_name="supermarket_retail_receipt.jpg", mime="image/jpeg")

            uploaded_file = st.file_uploader("Upload PDF or Image", type=['pdf', 'jpg', 'jpeg', 'png'])
            if uploaded_file:
                saved = coordinator.doc_manager.save_upload(uploaded_file.name, uploaded_file.read())
                target_path = saved["file_path"]

    with col_preview:
        if target_path and os.path.exists(target_path):
            if target_path.lower().endswith('.pdf'):
                from document.image_enhancer import PDFProcessor
                preview_imgs = PDFProcessor.convert_pdf_to_images(
                    target_path,
                    os.path.join(PROJECT_ROOT, "data", "processed", f"preview_{os.path.splitext(os.path.basename(target_path))[0]}")
                )
                if preview_imgs and os.path.exists(preview_imgs[0]):
                    st.image(preview_imgs[0], caption=f"PDF Preview (Page 1): {os.path.basename(target_path)}", use_container_width=True)
                else:
                    st.info(f"📄 **PDF Document Uploaded:** `{os.path.basename(target_path)}`")
            else:
                st.image(target_path, caption=f"Document Preview: {os.path.basename(target_path)}", use_container_width=True)

    if target_path and os.path.exists(target_path):
        if st.button("🚀 Run End-to-End Multimodal Analysis", type="primary"):
            with st.spinner("Processing through NLPL Part 1 -> DLL Engine -> Multimodal Fusion -> NLPL Part 2..."):
                result = coordinator.process_document(target_path)
                st.session_state["current_doc"] = result
                st.success("Document analyzed, validated, and persisted to database!")

    if "current_doc" in st.session_state:
        doc = st.session_state["current_doc"]
        st.divider()
        
        # Summary & Classification Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Document ID", doc["document_id"])
        c2.metric("Document Category", doc["document_type"].upper())
        c3.metric("Multimodal Confidence", f"{doc['confidence'] * 100:.1f}%")
        c4.metric("Validation Quality", f"{doc['validation_results']['quality_score']}%")
        
        st.info(f"📝 **Executive Summary:** {doc.get('summary', 'Summary not generated.')}")

        # Multi-Page PDF & YOLO Region Detection
        st.subheader("🔍 Visual Layout & YOLO Region Detection (DLL Modules III & V)")
        
        is_pdf_doc = doc["file_path"].lower().endswith(".pdf")
        pdf_pages = []
        if is_pdf_doc:
            from document.image_enhancer import PDFProcessor
            pdf_page_count = PDFProcessor.get_pdf_page_count(doc["file_path"])
            st.caption(f"📄 Multi-page PDF detected with **{pdf_page_count}** page(s).")
            rendered_pages = PDFProcessor.convert_pdf_to_images(doc["file_path"], f"data/processed/{doc['document_id']}_pages")
            if rendered_pages:
                selected_page_idx = 0
                if len(rendered_pages) > 1:
                    selected_page_idx = st.selectbox(
                        "Select PDF Page to View:",
                        options=range(len(rendered_pages)),
                        format_func=lambda i: f"Page {i+1} of {len(rendered_pages)}"
                    )
                target_visual_path = rendered_pages[selected_page_idx]
            else:
                target_visual_path = doc["file_path"]
        else:
            target_visual_path = doc["file_path"]

        annotated_out = f"data/processed/{doc['document_id']}_yolo.jpg"
        detection_res = region_detector.detect_regions(target_visual_path, annotated_out)
        
        det_col1, det_col2 = st.columns([1.2, 1])
        with det_col1:
            if os.path.exists(annotated_out):
                st.image(annotated_out, caption=f"YOLO Structural Bounding Box Layout ({os.path.basename(target_visual_path)})", use_container_width=True)
            elif os.path.exists(target_visual_path):
                st.image(target_visual_path, caption="Document Preview", use_container_width=True)
        with det_col2:
            st.write(f"**Detected Regions Count:** `{detection_res['detection_count']}`")
            for d in detection_res["detections"]:
                st.markdown(f"• **{d['label']}** — Conf: `{d['confidence']*100:.0f}%` — Box: `{d['box']}`")

        # Extracted Fields Table
        st.subheader("📋 Structured Information Extraction (NLPL Part 2)")
        fields = doc["extracted_data"]["fields"]
        
        table_rows = []
        for f_name, f_val in fields.items():
            if isinstance(f_val, dict) and "value" in f_val:
                val = f_val["value"]
                if isinstance(val, list):
                    val = f"{len(val)} entries (view JSON)"
                table_rows.append({
                    "Field Name": f_name.replace("_", " ").title(),
                    "Extracted Value": str(val),
                    "Confidence": f"{f_val.get('confidence', 0.0)*100:.1f}%",
                    "Confidence Level": f_val.get("level", "High"),
                    "Extraction Rule": f_val.get("extraction_method", "Regex/NER")
                })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # Validation Details
        st.subheader("🛡️ Validation & Arithmetic Rules")
        v_col1, v_col2 = st.columns([1, 1])
        with v_col1:
            for check in doc["validation_results"]["checks"]:
                icon = "✅" if check["status"] == "PASSED" else ("⚠️" if check["status"] == "WARNING" else "❌")
                st.write(f"{icon} **{check['rule']}:** {check['detail']}")
        with v_col2:
            st.json(doc["extracted_data"])

        # Multi-Turn Document Chatbot with Memory & In-Chat Export
        st.subheader("💬 Multi-Turn Document Chatbot (Groq LLM Acceleration)")
        st.caption("Powered by Groq Cloud API (`openai/gpt-oss-120b`). Ask contextual questions or command exports like *'Export this as Excel'*.")

        doc_id = doc["document_id"]
        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = {}
        if doc_id not in st.session_state.chat_sessions:
            st.session_state.chat_sessions[doc_id] = [
                {"role": "assistant", "content": f"Hello! I have analyzed **{doc['original_filename']}** ({doc['document_type'].upper()}). How can I assist you with this document?", "source": "System"}
            ]

        # Chat Controls
        chat_header_col1, chat_header_col2 = st.columns([4, 1])
        with chat_header_col2:
            if st.button("🗑️ Clear Chat", key=f"clear_chat_{doc_id}"):
                st.session_state.chat_sessions[doc_id] = [
                    {"role": "assistant", "content": f"Chat history cleared for **{doc['original_filename']}**. Ask me anything!", "source": "System"}
                ]
                st.rerun()

        # Render Chat History
        chat_container = st.container()
        with chat_container:
            for msg_idx, msg in enumerate(st.session_state.chat_sessions[doc_id]):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if "source" in msg and msg["source"] != "System":
                        st.caption(f"Confidence: {msg.get('confidence', 0.99)*100:.0f}% | Source: `{msg['source']}`")
                    
                    # If this response was an export intent, render the direct download button in chat
                    if "export_intent" in msg:
                        exp_fmt = msg["export_intent"]["format"]
                        if exp_fmt == "excel":
                            x_path = coordinator.export_manager.export_excel(doc['document_id'], doc)
                            with open(x_path, "rb") as f:
                                st.download_button(f"📥 Download {doc['document_id']}.xlsx", f.read(), file_name=f"{doc['document_id']}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"chat_dl_xlsx_{msg_idx}")
                        elif exp_fmt == "csv":
                            c_path = coordinator.export_manager.export_csv(doc['document_id'], doc)
                            with open(c_path, "rb") as f:
                                st.download_button(f"📥 Download {doc['document_id']}.csv", f.read(), file_name=f"{doc['document_id']}.csv", mime="text/csv", key=f"chat_dl_csv_{msg_idx}")
                        elif exp_fmt == "json":
                            st.download_button(f"📥 Download {doc['document_id']}.json", json.dumps(doc, indent=2), file_name=f"{doc['document_id']}.json", mime="application/json", key=f"chat_dl_json_{msg_idx}")

        # Quick Question Buttons
        st.write("**Quick Inquiries:**")
        q_cols = st.columns(4)
        quick_prompts = [
            "What is the total amount?",
            "Who is the customer / person?",
            "Export this document as Excel",
            "Give me a detailed summary"
        ]
        clicked_prompt = None
        for i, prompt_text in enumerate(quick_prompts):
            if q_cols[i].button(prompt_text, key=f"quick_{doc_id}_{i}"):
                clicked_prompt = prompt_text

        # Chat Input Bar
        user_input = st.chat_input("Ask a question, follow-up, or request an export...")
        prompt_to_send = user_input or clicked_prompt

        if prompt_to_send:
            # Append user message
            st.session_state.chat_sessions[doc_id].append({"role": "user", "content": prompt_to_send})
            
            # Query Chatbot with history
            history_turns = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_sessions[doc_id][:-1]
            ]
            bot_reply = coordinator.chatbot.answer_question(
                prompt_to_send,
                doc["extracted_data"],
                raw_text=doc.get("ocr_text", ""),
                history=history_turns
            )
            
            # Append assistant message
            assistant_msg = {
                "role": "assistant",
                "content": bot_reply["answer"],
                "confidence": bot_reply.get("confidence", 0.99),
                "source": bot_reply.get("source_field", "Groq LLM")
            }
            if "export_intent" in bot_reply:
                assistant_msg["export_intent"] = bot_reply["export_intent"]

            st.session_state.chat_sessions[doc_id].append(assistant_msg)
            st.rerun()

        # Exports
        st.subheader("📥 Export Center (Manual Downloads)")
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            json_str = json.dumps(doc, indent=2)
            st.download_button("Download JSON", json_str, file_name=f"{doc['document_id']}.json", mime="application/json", key=f"man_dl_json_{doc_id}")
        with exp_col2:
            csv_path = coordinator.export_manager.export_csv(doc['document_id'], doc)
            with open(csv_path, "rb") as f:
                st.download_button("Download CSV", f.read(), file_name=f"{doc['document_id']}.csv", mime="text/csv", key=f"man_dl_csv_{doc_id}")
        with exp_col3:
            xlsx_path = coordinator.export_manager.export_excel(doc['document_id'], doc)
            with open(xlsx_path, "rb") as f:
                st.download_button("Download Excel (.xlsx)", f.read(), file_name=f"{doc['document_id']}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"man_dl_xlsx_{doc_id}")

# ==========================================
# TAB 2: NLPL Laboratory
# ==========================================
with tabs[1]:
    st.subheader("🟢 Project 1 — Natural Language Processing Laboratory (Course Code: 2604734)")
    st.write("Complete coverage of NLPL Modules I–V: Preprocessing, BoW/TF-IDF, N-grams, NER, POS, Word2Vec/GloVe, and Social Media Sentiment.")
    
    nlp_tab1, nlp_tab2, nlp_tab3, nlp_tab4, nlp_tab5 = st.tabs([
        "1. Preprocessing (Module I)", 
        "2. BoW vs. TF-IDF & N-Grams (Module II)", 
        "3. NER & POS Tagging (Module II)", 
        "4. Word2Vec & GloVe Classifier (Module III)",
        "5. Social Media Sentiment (Module III)"
    ])
    
    with nlp_tab1:
        st.markdown("#### Text Cleaning, Stop-words, Stemming & Lemmatization (Sessions 1–8)")
        sample_text_nlp = st.text_area("Sample Document Text:", "The customer Rahul Shah paid INR 90,270 for cloud software licenses and engineering services on 15 August 2026.")
        if st.button("Run Text Preprocessing"):
            clean_res = coordinator.nlp_pipeline.cleaner.process(sample_text_nlp)
            st.write(f"**Raw Tokens ({len(clean_res['raw_tokens'])}):**", clean_res["raw_tokens"])
            st.write(f"**Cleaned Tokens ({len(clean_res['cleaned_tokens'])}):**", clean_res["cleaned_tokens"])
            
            comp_df = pd.DataFrame({
                "Cleaned Token": clean_res["cleaned_tokens"],
                "Porter Stemming": clean_res["stemmed_tokens"],
                "WordNet Lemmatization": clean_res["lemmatized_tokens"]
            })
            st.dataframe(comp_df, use_container_width=True)

    with nlp_tab2:
        st.markdown("#### BoW vs. TF-IDF Comparison & N-Gram Modeling (Sessions 9–14)")
        exp1_res = run_bow_tfidf_experiment()
        st.info(f"💡 **Academic Conclusion:** {exp1_res['academic_conclusion']}")
        st.dataframe(pd.DataFrame(exp1_res["comparison_matrix"]), use_container_width=True)
        
        st.markdown("##### N-Gram Frequency Distributions (Unigram, Bigram, Trigram)")
        exp2_res = run_ngram_experiment()
        col_b, col_t = st.columns(2)
        with col_b:
            st.write("**Top Bigrams:**")
            st.dataframe(pd.DataFrame(exp2_res["results"]["bigrams"]), use_container_width=True)
        with col_t:
            st.write("**Top Trigrams:**")
            st.dataframe(pd.DataFrame(exp2_res["results"]["trigrams"]), use_container_width=True)

    with nlp_tab3:
        st.markdown("#### Named Entity Recognition (NER) & POS Syntactic Tagging (Sessions 15–16)")
        ner_sample = "Rahul Shah paid INR 76,500 to TechFlow Solutions Pvt Ltd with GSTIN 24ABCDE1234F1Z5 on 15 August 2026."
        st.write(f"**Target Sentence:** `{ner_sample}`")
        entities = coordinator.nlp_pipeline.ner_extractor.extract_entities(ner_sample)
        st.dataframe(pd.DataFrame(entities), use_container_width=True)

    with nlp_tab4:
        st.markdown("#### Word2Vec & GloVe Text Classification (Sessions 17–21)")
        w2v_exp = run_word_vector_classifier_experiment()
        st.write("**Word2Vec Evaluation:**")
        st.dataframe(pd.DataFrame(w2v_exp["word2vec_evaluations"]), use_container_width=True)
        st.write("**GloVe Evaluation:**")
        st.dataframe(pd.DataFrame(w2v_exp["glove_evaluations"]), use_container_width=True)
        
        st.markdown("##### Cosine Similarity Explorer")
        w1 = st.text_input("Word 1:", "invoice")
        w2 = st.text_input("Word 2:", "bill")
        if st.button("Calculate Vector Similarity"):
            sim = embed_engine.compute_similarity(w1, w2)
            st.metric("Cosine Similarity Score", f"{sim:.4f}")

    with nlp_tab5:
        st.markdown("#### Social Media Sentiment Analysis (Sessions 22–24)")
        social_res = run_social_media_sentiment_experiment()
        st.dataframe(pd.DataFrame(social_res["samples_analyzed"]), use_container_width=True)

# ==========================================
# TAB 3: DLL Laboratory
# ==========================================
with tabs[2]:
    st.subheader("🔵 Project 2 — Deep Learning Laboratory (Course Code: 2604733)")
    st.write("Complete coverage of DLL Modules I–V: Neural Networks, Backprop, MNIST, Autoencoders, VAE, CAE, GAN, CNN, YOLO, and Transfer Learning.")
    
    dl_tab1, dl_tab2, dl_tab3, dl_tab4, dl_tab5 = st.tabs([
        "1. MLP & Backpropagation (Module I)", 
        "2. MNIST Digit Recognition (Module I & III)", 
        "3. Autoencoder, VAE & CAE (Module II)", 
        "4. GAN (Module II)", 
        "5. Transfer Learning & YOLO (Module III, IV, V)"
    ])
    
    with dl_tab1:
        st.markdown("#### Multilayer Perceptron (MLP) Backpropagation Engine (Sessions 1–6)")
        epochs_slider = st.slider("Number of Training Epochs:", 5, 50, 15)
        if st.button("Train MLP Model"):
            mlp_res = run_mlp_backprop_experiment(epochs=epochs_slider)
            st.metric("Final Model Accuracy", f"{mlp_res['accuracy']}%")
            st.metric("Loss Reduction", f"{mlp_res['initial_loss']:.4f} ➔ {mlp_res['final_loss']:.4f}")
            st.line_chart(pd.DataFrame(mlp_res["loss_history"]).set_index("epoch"))

    with dl_tab2:
        st.markdown("#### MNIST Handwritten Digit Classifier (Sessions 7–8, 21–22)")
        digit_sel = st.selectbox("Select Test Digit Pattern:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=7)
        if st.button("Run Digit Inference"):
            m_res = run_mnist_experiment(digit_sel)
            st.success(f"Predicted Digit: **{m_res['predicted_digit']}** (Confidence: {m_res['confidence']*100:.1f}%)")
            st.bar_chart(m_res["all_digit_probabilities"])

    with dl_tab3:
        st.markdown("#### Autoencoder Variants: Standard, VAE & CAE (Sessions 9–14)")
        ae_col1, ae_col2, ae_col3 = st.columns(3)
        with ae_col1:
            st.write("**Standard Autoencoder:**")
            st.json(run_autoencoder_experiment())
        with ae_col2:
            st.write("**Variational Autoencoder (VAE):**")
            st.json(run_vae_experiment())
        with ae_col3:
            st.write("**Convolutional Autoencoder (CAE):**")
            st.json(run_cae_experiment())

    with dl_tab4:
        st.markdown("#### Generative Adversarial Network (GAN) (Sessions 15–16)")
        gan_res = run_gan_experiment()
        st.json(gan_res)

    with dl_tab5:
        st.markdown("#### Transfer Learning (VGG16 vs. ResNet50) & YOLO (Sessions 23–40)")
        st.write("**Transfer Learning Comparison Benchmark:**")
        st.dataframe(pd.DataFrame(transfer_evaluator.compare_vgg_vs_resnet()), use_container_width=True)
        st.write("**Sequential LSTM & Neural Sentiment Analysis:**")
        st.json(run_sentiment_rnn_experiment())

# ==========================================
# TAB 4: Architecture & Syllabus Mapping
# ==========================================
with tabs[3]:
    st.subheader("📚 Full 40-Session Academic Syllabus Mapping (GLS FET B.Tech Semester VII)")
    
    col_map1, col_map2 = st.columns(2)
    with col_map1:
        st.markdown("""
        ### 🟢 NLPL (Course Code: 2604734) — 40 Sessions
        - **Sessions 1–8 (Module I):** Text Preprocessing, Tokenization, Stop-word removal, Punctuation removal, Stemming & Lemmatization (`nlp.preprocessing.cleaner`)
        - **Sessions 9–10 (Module II):** Bag of Words & TF-IDF vectorization & comparison (`nlp.vectorization.bow_tfidf`)
        - **Sessions 11–14 (Module II):** N-Gram modeling (Unigrams, Bigrams, Trigrams) across sentences (`nlp.ngrams.ngram_analyzer`)
        - **Sessions 15–16 (Module II):** Named Entity Recognition & POS tagging (`nlp.ner.ner_extractor`, `nlp.pos.pos_tagger`)
        - **Sessions 17–21 (Module III):** Text classification using Word2Vec & GloVe (`experiments.nlpl.exp4_word2vec_glove_classifier`)
        - **Sessions 22–24 (Module III):** Social media sentiment analysis (`experiments.nlpl.exp3_social_media_sentiment`)
        - **Sessions 25–28 (Module IV):** Pre-trained models (BERT/GPT) (`nlp.pretrained.transformer_model`)
        - **Sessions 29–32 (Module IV):** Advanced syntactic & semantic analysis (`fusion.multimodal_fusion`)
        - **Sessions 33–36 (Module V):** Text generation / chatbot (`chatbot.qa_engine`) & summarization (`summarization.summarizer`)
        - **Sessions 37–40 (Module V):** Real-time REST API Deployment (`app.main`)
        """)
        
    with col_map2:
        st.markdown("""
        ### 🔵 DLL (Course Code: 2604733) — 40 Sessions
        - **Sessions 1–4 (Module I):** Neural Networks, Multilayer Perceptron (`deep_learning.neural_network.mlp`)
        - **Sessions 5–6 (Module I):** Feedforward & Backpropagation algorithm (`experiments.dll.exp1_mlp_backprop`)
        - **Sessions 7–8 (Module I):** Digit classification with MNIST (`experiments.dll.exp2_mnist`)
        - **Sessions 9–10 (Module II):** Autoencoders image compression (`experiments.dll.exp3_autoencoder`)
        - **Sessions 11–12 (Module II):** Variational Autoencoders (VAE) (`experiments.dll.exp4_vae`)
        - **Sessions 13–14 (Module II):** Convolutional Autoencoders (CAE) document denoising (`experiments.dll.exp5_cae`)
        - **Sessions 15–16 (Module II):** Generative Adversarial Networks (GAN) (`experiments.dll.exp6_gan`)
        - **Sessions 17–22 (Module III):** CNN Digit & Document classification (`deep_learning.cnn.document_classifier`)
        - **Sessions 23–24 (Module III):** Object / Region Detection (`deep_learning.object_detection.yolo_detector`)
        - **Sessions 25–29 (Module IV):** RNN & LSTM sequential networks (`deep_learning.lstm.sequence_classifier`)
        - **Sessions 30–32 (Module IV):** Neural sentiment analysis (`experiments.dll.exp7_sentiment_rnn`)
        - **Sessions 33–34 (Module V):** Transfer Learning with YOLO (`deep_learning.object_detection.yolo_detector`)
        - **Sessions 35–37 (Module V):** Transfer Learning with VGG16 (`deep_learning.transfer_learning.transfer_model`)
        - **Sessions 38–40 (Module V):** Transfer Learning with ResNet50 (`deep_learning.transfer_learning.transfer_model`)
        """)

    st.success("🎯 **Academic Guarantee:** 100% alignment with GLS University FET B.Tech (CS&E) Semester VII curricula for both DLL (2604733) and NLPL (2604734).")

    # Expandable Citations
    with st.expander("📖 Foundational Literature & Research Citations", expanded=False):
        st.markdown("""
        * **CNN Document Recognition:** LeCun et al. (1998) *Gradient-based learning applied to document recognition.*
        * **ResNet Architecture:** He et al. (2016) *Deep residual learning for image recognition.* IEEE CVPR.
        * **Bi-LSTM Tagging:** Graves & Schmidhuber (2005) *Framewise phoneme classification with bidirectional LSTM.*
        * **BERT Transformers:** Devlin et al. (2018) *BERT: Pre-training of deep bidirectional transformers.*
        * **Variational Autoencoders:** Kingma & Welling (2013) *Auto-encoding variational Bayes.*
        * **YOLO Region Detection:** Redmon et al. (2016) *You only look once: Unified real-time object detection.*
        """)
