"""
NLPL Experiment 3: Sentiment Analysis on Social Media Text (Sessions 22-24)
Direct mapping to syllabus: Classifies social media reviews, customer posts, and tweets into Positive, Negative, and Neutral with polarity ratings.
"""

from deep_learning.lstm.sequence_classifier import SequenceAndSentimentAnalyzer

def run_social_media_sentiment_experiment() -> dict:
    analyzer = SequenceAndSentimentAnalyzer()
    
    tweets_and_posts = [
        {"post": "Amazing experience with TechFlow! Cloud billing integration was seamless and support was super responsive. #happycustomer", "author": "@sarah_tech"},
        {"post": "Extremely frustrated with unexpected tax surcharge and failed payment on order #482. Fix this ASAP! #billingfail", "author": "@alex_dev"},
        {"post": "Standard monthly service invoice INV-1023 received. Processing payment as per vendor agreement.", "author": "@finance_corp"},
        {"post": "Big shoutout to @ananya_ai for architecting the deep learning OCR pipeline. Unbelievable accuracy! 🚀🔥", "author": "@data_lead"}
    ]
    
    analyzed = []
    for item in tweets_and_posts:
        res = analyzer.analyze_sentiment(item["post"])
        analyzed.append({
            "author": item["author"],
            "post_text": item["post"],
            "classified_sentiment": res["sentiment"],
            "polarity_score": res["polarity_score"],
            "confidence": f"{res['confidence']*100:.0f}%"
        })
        
    return {
        "experiment_name": "Social Media Sentiment Analysis",
        "syllabus_mapping": "NLPL Module III (Sessions 22-24)",
        "samples_analyzed": analyzed,
        "sentiment_distribution": {
            "Positive": sum(1 for a in analyzed if a["classified_sentiment"] == "Positive"),
            "Negative": sum(1 for a in analyzed if "Negative" in a["classified_sentiment"]),
            "Neutral": sum(1 for a in analyzed if "Neutral" in a["classified_sentiment"])
        },
        "status": "COMPLETED"
    }

if __name__ == "__main__":
    print(run_social_media_sentiment_experiment())
