# Academic References, Syllabus Alignment & Citations

**Multimodal Document Intelligence Platform**  
*Dual Projects: NLPL (Course 2604734) & DLL (Course 2604735) — Semester VII (CEC)*

---

## 1. Official Course Syllabus Mapping

### A. Deep Learning Lab (Course Code: 2604735)
| Module / Session | Topic Covered | Implementation in Codebase |
| :--- | :--- | :--- |
| **Module I (Sessions 1–6)** | Multi-Layer Perceptron (MLP) & Backpropagation | [`deep_learning/neural_network/mlp.py`](file:///c:/SEM%207%20CEC/deep_learning/neural_network/mlp.py)<br>[`experiments/dll/exp1_mlp_backprop.py`](file:///c:/SEM%207%20CEC/experiments/dll/exp1_mlp_backprop.py) |
| **Module II (Sessions 7–12)** | Handwritten Digit Recognition (MNIST) | [`experiments/dll/exp2_mnist.py`](file:///c:/SEM%207%20CEC/experiments/dll/exp2_mnist.py) |
| **Module III (Sessions 13–18)** | Document Layout CNN Classifier | [`deep_learning/cnn/document_classifier.py`](file:///c:/SEM%207%20CEC/deep_learning/cnn/document_classifier.py) |
| **Module IV (Sessions 19–24)** | Autoencoders, VAE & CAE Denoising | [`experiments/dll/exp3_autoencoder.py`](file:///c:/SEM%207%20CEC/experiments/dll/exp3_autoencoder.py)<br>[`experiments/dll/exp4_vae.py`](file:///c:/SEM%207%20CEC/experiments/dll/exp4_vae.py)<br>[`experiments/dll/exp5_cae.py`](file:///c:/SEM%207%20CEC/experiments/dll/exp5_cae.py) |
| **Module V (Sessions 25–30)** | Generative Adversarial Networks (GAN) | [`experiments/dll/exp6_gan.py`](file:///c:/SEM%207%20CEC/experiments/dll/exp6_gan.py) |
| **Module VI (Sessions 31–36)** | Sequence Tagging & Recurrent Models (Bi-LSTM / RNN) | [`deep_learning/lstm/sequence_classifier.py`](file:///c:/SEM%207%20CEC/deep_learning/lstm/sequence_classifier.py)<br>[`experiments/dll/exp7_sentiment_rnn.py`](file:///c:/SEM%207%20CEC/experiments/dll/exp7_sentiment_rnn.py) |
| **Module VII (Sessions 37–40)** | Transfer Learning (VGG16 / ResNet50) & YOLO Region Detection | [`deep_learning/transfer_learning/transfer_model.py`](file:///c:/SEM%207%20CEC/deep_learning/transfer_learning/transfer_model.py)<br>[`deep_learning/object_detection/yolo_detector.py`](file:///c:/SEM%207%20CEC/deep_learning/object_detection/yolo_detector.py) |

---

### B. Natural Language Processing Lab (Course Code: 2604734)
| Module / Session | Topic Covered | Implementation in Codebase |
| :--- | :--- | :--- |
| **Module I (Sessions 1–8)** | Text Cleaning, Stopwords, Stemming & Lemmatization | [`nlp/preprocessing/cleaner.py`](file:///c:/SEM%207%20CEC/nlp/preprocessing/cleaner.py)<br>[`nlp/pipeline.py`](file:///c:/SEM%207%20CEC/nlp/pipeline.py) |
| **Module II (Sessions 9–14)** | BoW vs. TF-IDF & N-Gram Statistical Language Modeling | [`nlp/vectorization/bow_tfidf.py`](file:///c:/SEM%207%20CEC/nlp/vectorization/bow_tfidf.py)<br>[`nlp/ngrams/ngram_model.py`](file:///c:/SEM%207%20CEC/nlp/ngrams/ngram_model.py)<br>[`experiments/nlpl/exp1_bow_tfidf_comparison.py`](file:///c:/SEM%207%20CEC/experiments/nlpl/exp1_bow_tfidf_comparison.py)<br>[`experiments/nlpl/exp2_ngram_modelling.py`](file:///c:/SEM%207%20CEC/experiments/nlpl/exp2_ngram_modelling.py) |
| **Module III (Sessions 15–20)** | Part-of-Speech (POS) Tagging & Named Entity Recognition (NER) | [`nlp/pos/pos_tagger.py`](file:///c:/SEM%207%20CEC/nlp/pos/pos_tagger.py)<br>[`nlp/ner/entity_extractor.py`](file:///c:/SEM%207%20CEC/nlp/ner/entity_extractor.py) |
| **Module IV (Sessions 21–28)** | Word Embeddings (Word2Vec, GloVe) & Social Media Sentiment | [`nlp/embeddings/word_embeddings.py`](file:///c:/SEM%207%20CEC/nlp/embeddings/word_embeddings.py)<br>[`experiments/nlpl/exp3_social_media_sentiment.py`](file:///c:/SEM%207%20CEC/experiments/nlpl/exp3_social_media_sentiment.py)<br>[`experiments/nlpl/exp4_word2vec_glove_classifier.py`](file:///c:/SEM%207%20CEC/experiments/nlpl/exp4_word2vec_glove_classifier.py) |
| **Module V (Sessions 29–36)** | Transformers (BERT) & Extractive Summarization | [`nlp/pretrained/bert_embedder.py`](file:///c:/SEM%207%20CEC/nlp/pretrained/bert_embedder.py)<br>[`summarization/summarizer.py`](file:///c:/SEM%207%20CEC/summarization/summarizer.py) |
| **Module VI (Sessions 37–40)** | Conversational Q&A Document Chatbot | [`chatbot/qa_engine.py`](file:///c:/SEM%207%20CEC/chatbot/qa_engine.py) |

---

## 2. Foundational Research Papers & Literature Citations

1. **Convolutional Neural Networks for Visual Document Layout:**
   * *LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).* "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*, 86(11), 2278-2324.
   * *He, K., Zhang, X., Ren, S., & Sun, J. (2016).* "Deep residual learning for image recognition." *IEEE CVPR*, 770-778.
2. **Sequential Entity Extraction & Recurrent Neural Networks:**
   * *Graves, A., & Schmidhuber, J. (2005).* "Framewise phoneme classification with bidirectional LSTM and other neural network architectures." *Neural Networks*, 18(5-6), 602-610.
   * *Hochreiter, S., & Schmidhuber, J. (1997).* "Long short-term memory." *Neural Computation*, 9(8), 1735-1780.
3. **Transformer Embeddings & Language Representation:**
   * *Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018).* "BERT: Pre-training of deep bidirectional transformers for language understanding." *arXiv:1810.04805*.
   * *Reimers, N., & Gurevych, I. (2019).* "Sentence-BERT: Sentence embeddings using Siamese BERT-networks." *EMNLP-IJCNLP*, 3982-3992.
4. **Variational Autoencoders & Generative Models:**
   * *Kingma, D. P., & Welling, M. (2013).* "Auto-encoding variational Bayes." *arXiv:1312.6114*.
   * *Goodfellow, I., et al. (2014).* "Generative adversarial nets." *NeurIPS*, 2672-2680.
5. **Word Embeddings & Vector Representations:**
   * *Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013).* "Efficient estimation of word representations in vector space." *arXiv:1301.3781*.
   * *Pennington, J., Socher, R., & Manning, C. D. (2014).* "GloVe: Global vectors for word representation." *EMNLP*, 1532-1543.
6. **Object & Region Detection in Documents:**
   * *Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).* "You only look once: Unified, real-time object detection." *IEEE CVPR*, 779-788.

---

## 3. Standard Benchmark Datasets Used / Referenced

1. **MNIST (Modified National Institute of Standards and Technology):**
   * 70,000 $28\times 28$ grayscale images of handwritten digits (0–9) for deep neural network validation.
2. **RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing):**
   * 400,000 grayscale document images across 16 categories used as benchmark standards for document layout categorization.
3. **SROIE (ICDAR 2019 Scanned Receipts OCR and Information Extraction):**
   * Scanned receipts for optical character recognition, bounding box text detection, and key information extraction.
4. **CoNLL-2003:**
   * Standard benchmark corpus for Named Entity Recognition (PERSON, ORG, LOC, MISC).
