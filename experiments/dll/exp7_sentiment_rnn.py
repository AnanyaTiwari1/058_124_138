"""
DLL Experiment 7: Neural Sentiment Analysis with RNN & LSTM
Demonstrates sequential text encoding for customer feedback and sentiment polarity.
"""

from deep_learning.lstm.sequence_classifier import SequenceAndSentimentAnalyzer

def run_sentiment_rnn_experiment() -> dict:
    analyzer = SequenceAndSentimentAnalyzer()
    
    samples = [
        "Thank you for the prompt delivery and premium technical support. Highly approved!",
        "Invoice payment is severely overdue with penalty charges applied. Please resolve immediately.",
        "Tax invoice generated for standard cloud storage services."
    ]
    
    results = []
    for s in samples:
        res = analyzer.analyze_sentiment(s)
        results.append({
            "text": s,
            "sentiment": res["sentiment"],
            "polarity": res["polarity_score"]
        })
        
    return {
        "experiment_name": "Neural Sentiment Analysis using Sequential RNN/LSTM",
        "sample_evaluations": results,
        "model_architecture": "BiLSTM(Embedding(64) -> BiLSTM(Hidden=64) -> Dense -> Softmax)",
        "status": "COMPLETED"
    }

if __name__ == "__main__":
    print(run_sentiment_rnn_experiment())
