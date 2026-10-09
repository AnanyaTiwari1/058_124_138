"""
NLPL Experiment 1: Bag of Words vs. TF-IDF Comparative Study
Direct syllabus mapping: Evaluates frequency vs. discriminative power across multi-document corpus.
"""

from nlp.vectorization.bow_tfidf import Vectorizer

def run_bow_tfidf_experiment() -> dict:
    corpus = [
        "invoice total amount tax due techflow solutions pvt ltd payment",
        "aroma cafe receipt coffee muffin sandwich breakfast payment total",
        "resume engineer machine learning python pytorch fastAPI deep learning",
        "government identity card name dob address india identity number"
    ]
    
    vectorizer = Vectorizer(corpus)
    test_tokens = ["payment", "invoice", "coffee", "total", "learning"]
    comparison = vectorizer.compare_bow_vs_tfidf(test_tokens)
    
    return {
        "experiment_name": "BoW vs. TF-IDF Effectiveness Comparison",
        "corpus_documents_count": len(corpus),
        "vocabulary_size": len(vectorizer.vocabulary),
        "comparison_matrix": comparison,
        "academic_conclusion": "BoW measures raw frequency; TF-IDF downweights common words and emphasizes distinctive domain keywords.",
        "status": "COMPLETED"
    }

if __name__ == "__main__":
    print(run_bow_tfidf_experiment())
