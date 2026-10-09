"""
Unit tests for DLL Core Engine and Experiments
"""

import os
import pytest
import numpy as np
from deep_learning.neural_network.mlp import MultilayerPerceptron
from deep_learning.cnn.document_classifier import DocumentImageClassifier
from deep_learning.transfer_learning.transfer_model import TransferLearningEvaluator
from deep_learning.lstm.sequence_classifier import SequenceAndSentimentAnalyzer
from experiments.dll.exp1_mlp_backprop import run_mlp_backprop_experiment
from experiments.dll.exp2_mnist import run_mnist_experiment
from experiments.dll.exp3_autoencoder import run_autoencoder_experiment
from experiments.dll.exp4_vae import run_vae_experiment
from experiments.dll.exp5_cae import run_cae_experiment
from experiments.dll.exp6_gan import run_gan_experiment
from experiments.dll.exp7_sentiment_rnn import run_sentiment_rnn_experiment

def test_mlp_and_backprop():
    res = run_mlp_backprop_experiment(epochs=10)
    assert res["status"] == "COMPLETED"
    assert res["final_loss"] <= res["initial_loss"]

def test_cnn_classifier():
    classifier = DocumentImageClassifier()
    sample_img = "data/sample_documents/invoice_sample.jpg"
    if os.path.exists(sample_img):
        pred = classifier.predict(sample_img)
        assert pred["predicted_class"] == "invoice"
        assert pred["confidence"] > 0.8

def test_transfer_learning():
    evaluator = TransferLearningEvaluator()
    resnet_eval = evaluator.evaluate_model("ResNet50")
    assert resnet_eval["metrics"]["accuracy"] > 0.95
    comp = evaluator.compare_vgg_vs_resnet()
    assert len(comp) == 4

def test_lstm_and_sentiment():
    analyzer = SequenceAndSentimentAnalyzer()
    sent = analyzer.analyze_sentiment("Thank you, approved and paid in full!")
    assert sent["sentiment"] == "Positive"
    seq = analyzer.predict_sequence_class(["invoice", "tax", "total", "amount"])
    assert seq["predicted_class"] in ['invoice', 'receipt', 'resume', 'id_card']

def test_autoencoders_and_gan():
    assert run_autoencoder_experiment()["status"] == "COMPLETED"
    assert run_vae_experiment()["status"] == "COMPLETED"
    assert run_cae_experiment()["status"] == "COMPLETED"
    assert run_gan_experiment()["status"] == "COMPLETED"
    assert run_mnist_experiment(7)["predicted_digit"] == 7
    assert run_sentiment_rnn_experiment()["status"] == "COMPLETED"
